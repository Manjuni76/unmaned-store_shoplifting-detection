"""
Multi-Part Attention-based Classifier
부위별 Feature를 Attention으로 결합하여 분류
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartAttentionClassifier(nn.Module):
    """
    5개 부위별 Feature를 Projection → Attention → Classification
    
    Args:
        part_dims: 부위별 입력 차원 딕셔너리 {'head': 360, 'arms': 432, ...}
        embed_dim: 통일된 임베딩 차원 (default: 256)
        num_heads: Attention head 개수 (default: 8)
        num_encoder_layers: Transformer 인코더 레이어 수 (default: 2)
        dropout: Dropout 비율 (default: 0.3)
        num_classes: 분류 클래스 수 (default: 2)
    """
    def __init__(self, 
                 part_dims,
                 embed_dim=256, 
                 num_heads=8, 
                 num_encoder_layers=2,
                 dropout=0.3,
                 num_classes=2):
        super(PartAttentionClassifier, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_parts = len(part_dims)
        self.part_names = list(part_dims.keys())
        
        # 1. 부위별 Projection Layers (차원 통일)
        self.projections = nn.ModuleDict()
        for part_name, input_dim in part_dims.items():
            self.projections[part_name] = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)  # Projection에는 낮은 dropout
            )
        
        # 2. Positional Encoding (부위별 위치 정보)
        # 학습 가능한 위치 임베딩
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_parts, embed_dim) * 0.02
        )
        
        # 3. Transformer Encoder (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # FFN 차원
            dropout=dropout,
            activation='gelu',  # GELU activation
            batch_first=True,
            norm_first=True  # Pre-LN (더 안정적)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 4. Classification Head
        # [CLS] 토큰 방식 대신 모든 부위 평균(Global Average Pooling)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, part_features_dict):
        """
        Args:
            part_features_dict: {'head': tensor(B, 360), 'arms': tensor(B, 432), ...}
        
        Returns:
            logits: (B, num_classes)
        """
        batch_size = next(iter(part_features_dict.values())).size(0)
        
        # 1. Projection: 각 부위를 embed_dim으로 통일
        projected_parts = []
        for part_name in self.part_names:
            feat = part_features_dict[part_name]  # (B, part_dim)
            proj_feat = self.projections[part_name](feat)  # (B, embed_dim)
            projected_parts.append(proj_feat)
        
        # 2. Stack: (B, num_parts, embed_dim)
        x = torch.stack(projected_parts, dim=1)
        
        # 3. Positional Encoding 추가
        x = x + self.pos_embedding
        
        # 4. Transformer Encoder (Self-Attention)
        # 부위 간 상호작용 학습
        attn_output = self.transformer_encoder(x)  # (B, num_parts, embed_dim)
        
        # 5. Global Average Pooling
        # 모든 부위의 정보를 평균으로 통합
        pooled = attn_output.mean(dim=1)  # (B, embed_dim)
        
        # 6. Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits
    
    def get_attention_weights(self, part_features_dict):
        """
        어텐션 가중치 시각화를 위한 메서드
        (디버깅/분석용)
        """
        batch_size = next(iter(part_features_dict.values())).size(0)
        
        # Projection
        projected_parts = []
        for part_name in self.part_names:
            feat = part_features_dict[part_name]
            proj_feat = self.projections[part_name](feat)
            projected_parts.append(proj_feat)
        
        x = torch.stack(projected_parts, dim=1)
        x = x + self.pos_embedding
        
        # Attention weights 추출은 별도 구현 필요
        # (실제로는 TransformerEncoder의 각 레이어에서 attention_weights를 받아야 함)
        return None  # TODO: 필요시 구현


def create_attention_classifier(stg_nf_models_dict, sample_data_dict, num_classes=2, 
                                embed_dim=256, num_heads=8, num_encoder_layers=2,
                                dropout=0.3, device=0):
    """
    Attention 기반 분류기 생성 (STG-NF Feature Extractor + Attention Classifier)
    
    Args:
        stg_nf_models_dict: 부위별 STG-NF 모델 딕셔너리
        sample_data_dict: 샘플 데이터 딕셔너리 (feature 차원 계산용)
        num_classes: 분류 클래스 수
        embed_dim: 임베딩 차원
        num_heads: Attention head 개수
        num_encoder_layers: Transformer 레이어 수
        dropout: Dropout 비율
        device: 디바이스
    
    Returns:
        model: Multi_STG_NF_with_Attention 모델
    """
    from model_builder import Multi_STG_NF_with_Attention
    
    # 부위별 Feature 차원 계산
    part_dims = {}
    with torch.no_grad():
        for part_name, stg_nf_model in stg_nf_models_dict.items():
            # ActNorm 초기화를 위해 train 모드로 변경
            stg_nf_model.train()
            
            sample_input = sample_data_dict[part_name].to(device)
            output = stg_nf_model(sample_input)
            
            # STG-NF는 tuple (z, nll) 반환 - 첫 번째 요소가 feature
            if isinstance(output, tuple):
                feature = output[0]
            else:
                feature = output
            
            feature_dim = feature.view(feature.size(0), -1).size(1)
            part_dims[part_name] = feature_dim
            print(f"  [{part_name}] Feature dim: {feature_dim}")
            
            # 다시 eval 모드로 변경
            stg_nf_model.eval()
    
    # Attention Classifier 생성
    attention_classifier = PartAttentionClassifier(
        part_dims=part_dims,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        num_classes=num_classes
    ).to(device)
    
    # 전체 모델 (STG-NF + Attention Classifier)
    model = Multi_STG_NF_with_Attention(
        stg_nf_models_dict=stg_nf_models_dict,
        attention_classifier=attention_classifier,
        device=device
    ).to(device)
    
    return model
