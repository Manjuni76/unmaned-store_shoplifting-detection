"""
Anomaly Score + Feature Attention Classifier
STG-NF의 Feature Vector와 Anomaly Score(Log-Likelihood)를 결합한 분류기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyScoreAttentionClassifier(nn.Module):
    """
    5개 부위별 Feature + Anomaly Score를 결합하여 Attention으로 분류
    
    핵심 아이디어:
    - Feature Vector (128차원): 동작의 '형태' 표현
    - Anomaly Score (1차원): '얼마나 비정상인가' 표현
    - 두 정보를 결합하여 도난 행동 탐지 성능 향상
    
    Args:
        part_dims: 부위별 입력 차원 딕셔너리 {'head': 360, 'arms': 432, ...}
        embed_dim: 통일된 임베딩 차원 (default: 256)
        score_embed_dim: Anomaly score 임베딩 차원 (default: 16)
        num_heads: Attention head 개수 (default: 8)
        num_encoder_layers: Transformer 인코더 레이어 수 (default: 2)
        dropout: Dropout 비율 (default: 0.3)
        num_classes: 분류 클래스 수 (default: 2)
    """
    def __init__(self, 
                 part_dims,
                 embed_dim=256, 
                 score_embed_dim=16,
                 num_heads=8, 
                 num_encoder_layers=2,
                 dropout=0.3,
                 num_classes=2):
        super(AnomalyScoreAttentionClassifier, self).__init__()
        
        self.embed_dim = embed_dim
        self.score_embed_dim = score_embed_dim
        self.num_parts = len(part_dims)
        self.part_names = list(part_dims.keys())
        
        # Feature 임베딩 차원 (score 임베딩 공간 확보)
        self.feat_embed_dim = embed_dim - score_embed_dim
        
        # 1. 부위별 Feature Projection Layers (차원 통일)
        self.feat_projections = nn.ModuleDict()
        for part_name, input_dim in part_dims.items():
            self.feat_projections[part_name] = nn.Sequential(
                nn.Linear(input_dim, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
        
        # 2. Anomaly Score Projection (1차원 → score_embed_dim)
        # 각 부위별로 독립적인 score embedding
        self.score_projections = nn.ModuleDict()
        for part_name in part_dims.keys():
            self.score_projections[part_name] = nn.Sequential(
                nn.Linear(1, score_embed_dim),
                nn.LayerNorm(score_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
        
        # 3. Positional Encoding (부위별 위치 정보)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_parts, embed_dim) * 0.02
        )
        
        # 4. Transformer Encoder (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 5. Classification Head
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
    
    def forward(self, part_features_dict, part_scores_dict):
        """
        Args:
            part_features_dict: {'head': tensor(B, 360), 'arms': tensor(B, 432), ...}
            part_scores_dict: {'head': tensor(B, 1), 'arms': tensor(B, 1), ...}
                             (Anomaly Score = Negative Log-Likelihood)
        
        Returns:
            logits: (B, num_classes)
        """
        batch_size = next(iter(part_features_dict.values())).size(0)
        
        # 1. Feature Projection + Score Projection + Concat
        combined_parts = []
        for part_name in self.part_names:
            # Feature embedding
            feat = part_features_dict[part_name]  # (B, part_dim)
            feat_emb = self.feat_projections[part_name](feat)  # (B, feat_embed_dim)
            
            # Score embedding
            score = part_scores_dict[part_name]  # (B,) or (B, 1)
            if score.dim() == 1:
                score = score.unsqueeze(1)  # (B, 1)
            score_emb = self.score_projections[part_name](score)  # (B, score_embed_dim)
            
            # Concatenate: (B, feat_embed_dim + score_embed_dim = embed_dim)
            combined = torch.cat([feat_emb, score_emb], dim=1)
            combined_parts.append(combined)
        
        # 2. Stack: (B, num_parts, embed_dim)
        x = torch.stack(combined_parts, dim=1)
        
        # 3. Positional Encoding 추가
        x = x + self.pos_embedding
        
        # 4. Transformer Encoder (Self-Attention)
        attn_output = self.transformer_encoder(x)  # (B, num_parts, embed_dim)
        
        # 5. Global Average Pooling
        pooled = attn_output.mean(dim=1)  # (B, embed_dim)
        
        # 6. Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits


class Multi_STG_NF_with_AnomalyScoreAttention(nn.Module):
    """
    전체 모델: STG-NF Feature Extractor + Anomaly Score + Attention Classifier
    """
    def __init__(self, stg_nf_models_dict, anomaly_attention_classifier, device='cpu'):
        super(Multi_STG_NF_with_AnomalyScoreAttention, self).__init__()
        
        self.device = device
        self.part_names = list(stg_nf_models_dict.keys())
        
        # STG-NF 모델들 (Freeze)
        self.stg_nf_models = nn.ModuleDict(stg_nf_models_dict)
        for model in self.stg_nf_models.values():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        
        # Anomaly Score Attention Classifier (Trainable)
        self.classifier = anomaly_attention_classifier
    
    def forward(self, data_dict):
        """
        Args:
            data_dict: {'head': (B, C, T, V), 'arms': (B, C, T, V), ...}
        
        Returns:
            logits: (B, num_classes)
        """
        part_features = {}
        part_scores = {}
        
        # 각 부위별 Feature + Anomaly Score 추출
        with torch.no_grad():
            for part_name in self.part_names:
                x = data_dict[part_name]
                
                # STG-NF forward: (z, log_det_J)
                # z: feature vector (B, C, T, V)
                # log_det_J: log-likelihood (B,)
                z, log_det_J = self.stg_nf_models[part_name](x)
                
                # Feature: flatten
                feat = z.view(z.size(0), -1)  # (B, C*T*V)
                part_features[part_name] = feat
                
                # Anomaly Score: -log_det_J (높을수록 비정상)
                # STG-NF는 정상 데이터로 학습했으므로, 
                # 정상 데이터는 높은 likelihood (낮은 -log_det_J)
                # 비정상 데이터는 낮은 likelihood (높은 -log_det_J)
                anomaly_score = -log_det_J  # (B,)
                part_scores[part_name] = anomaly_score
        
        # Attention Classifier
        logits = self.classifier(part_features, part_scores)
        
        return logits


def create_anomaly_score_attention_classifier(stg_nf_models_dict, sample_data_dict, 
                                              num_classes=2, 
                                              embed_dim=256, 
                                              score_embed_dim=16,
                                              num_heads=8, 
                                              num_encoder_layers=2,
                                              dropout=0.3, 
                                              device='cpu'):
    """
    Anomaly Score Attention 기반 분류기 생성
    
    Args:
        stg_nf_models_dict: 부위별 STG-NF 모델 딕셔너리
        sample_data_dict: 샘플 데이터 딕셔너리 (feature 차원 계산용)
        num_classes: 분류 클래스 수
        embed_dim: 전체 임베딩 차원
        score_embed_dim: Anomaly score 임베딩 차원
        num_heads: Attention head 개수
        num_encoder_layers: Transformer 레이어 수
        dropout: Dropout 비율
        device: 디바이스
    
    Returns:
        model: Multi_STG_NF_with_AnomalyScoreAttention 모델
    """
    # 부위별 Feature 차원 계산
    part_dims = {}
    print("\n[INFO] Feature 차원 계산:")
    with torch.no_grad():
        for part_name, stg_nf_model in stg_nf_models_dict.items():
            stg_nf_model.train()
            
            sample_input = sample_data_dict[part_name].to(device)
            z, log_det_J = stg_nf_model(sample_input)
            
            feature_dim = z.view(z.size(0), -1).size(1)
            part_dims[part_name] = feature_dim
            print(f"  [{part_name}] Feature dim: {feature_dim}, Score dim: 1")
            
            stg_nf_model.eval()
    
    print(f"\n[INFO] Embedding 구조:")
    print(f"  전체 임베딩 차원: {embed_dim}")
    print(f"  Feature 임베딩: {embed_dim - score_embed_dim}")
    print(f"  Score 임베딩: {score_embed_dim}")
    
    # Anomaly Score Attention Classifier 생성
    anomaly_attention_classifier = AnomalyScoreAttentionClassifier(
        part_dims=part_dims,
        embed_dim=embed_dim,
        score_embed_dim=score_embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        num_classes=num_classes
    ).to(device)
    
    # 전체 모델
    model = Multi_STG_NF_with_AnomalyScoreAttention(
        stg_nf_models_dict=stg_nf_models_dict,
        anomaly_attention_classifier=anomaly_attention_classifier,
        device=device
    ).to(device)
    
    return model
