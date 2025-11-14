"""모델 정의 파일 - STG-NF + Attention 분류기"""
import torch
import torch.nn as nn

try:
    from models.STG_NF.model_pose import STG_NF
except ImportError:
    print("[WARNING] STG-NF 모델 import 실패")
    STG_NF = None

def create_stgnf_model(in_channels=3, hidden_dim=64, num_layers=8, num_frames=12, 
                       num_joints=18, graph_cfg=None, device='cuda', subset_idx=None):
    if STG_NF is None:
        raise ImportError("STG-NF 모델을 import할 수 없습니다.")
    
    if graph_cfg is None:
        graph_cfg = {'layout': 'openpose', 'strategy': 'spatial', 'max_hop': 1}
    
    pose_shape = (in_channels, num_frames, num_joints)
    
    model = STG_NF(
        pose_shape=pose_shape, hidden_channels=hidden_dim, K=num_layers, L=3, R=2,
        actnorm_scale=1.0, flow_permutation='invconv', flow_coupling='affine',
        subset_idx=subset_idx, LU_decomposed=True, edge_importance=False,
        temporal_kernel_size=9, strategy=graph_cfg.get('strategy', 'spatial'),
        max_hops=graph_cfg.get('max_hop', 1), device=device, learn_top=False
    )
    
    return model.to(device)

class Multi_STG_NF_with_Attention(nn.Module):
    def __init__(self, stg_nf_models_dict, attention_classifier, device='cuda'):
        super().__init__()
        self.part_names = list(stg_nf_models_dict.keys())
        self.device = device
        self.stg_nf_models = nn.ModuleDict()
        
        for part_name, stg_nf_model in stg_nf_models_dict.items():
            for param in stg_nf_model.parameters():
                param.requires_grad = False
            stg_nf_model.eval()
            self.stg_nf_models[part_name] = stg_nf_model
        
        self.attention_classifier = attention_classifier
    
    def forward(self, x_dict):
        part_features = {}
        with torch.no_grad():
            for part_name in self.part_names:
                x = x_dict[part_name]
                output = self.stg_nf_models[part_name](x)
                feat = output[0] if isinstance(output, tuple) else output
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                part_features[part_name] = feat
        
        logits = self.attention_classifier(part_features)
        return logits
    
    def get_attention_weights(self, x_dict):
        part_features = {}
        with torch.no_grad():
            for part_name in self.part_names:
                x = x_dict[part_name]
                output = self.stg_nf_models[part_name](x)
                feat = output[0] if isinstance(output, tuple) else output
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                part_features[part_name] = feat
        return self.attention_classifier.get_attention_weights(part_features)
