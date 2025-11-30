"""STG-NF 체크포인트 로더 (Attention 학습용)
ActNorm 통계를 재초기화하지 않고 그대로 사용하도록 inited 플래그를 강제 설정한다.
"""

import os
import torch
from args import Config
from .model_builder import create_stgnf_model


def _mark_actnorm_initialized(model):
    #state_dict에 bias/logs가 로드된 ActNorm 모듈들의 inited 플래그를 True로 설정
    for name, module in model.named_modules():
        # _ActNorm은 bias/logs 속성을 가지며 inited 플래그가 존재
        if hasattr(module, 'bias') and hasattr(module, 'logs') and hasattr(module, 'inited'):
            # bias/logs가 학습된 값(0이 아닌) 또는 그냥 로드되었다면 초기화로 간주
            module.inited = True


def load_stgnf_model(part_name: str, joint_subset, device):
    # 단일 부위 STG-NF 모델 로드 (ActNorm 재초기화 없음)
    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, f'stgnf_{part_name}_fin.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"STG-NF 체크포인트 없음: {checkpoint_path}")

    num_joints = len(joint_subset) if joint_subset is not None else 18
    model = create_stgnf_model(
        in_channels=Config.STGNF.IN_CHANNELS,
        hidden_dim=Config.STGNF.HIDDEN_CHANNELS,
        num_layers=Config.STGNF.K,
        num_frames=Config.Data.SEG_LEN,
        num_joints=num_joints,
        graph_cfg=Config.STGNF.GRAPH_CFG,
        device=device,
        subset_idx=joint_subset
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    _mark_actnorm_initialized(model)
    model.eval()
    return model


def load_all_stgnf_models(device):
    # 모든 부위 STG-NF 모델 딕셔너리 로드
    models = {}
    for part in Config.Joint.BODY_PARTS:
        subset = Config.Joint.JOINT_SUBSET_MAP[part]
        models[part] = load_stgnf_model(part, subset, device)
        print(f"STG-NF {part} 모델 로드 완료")
    return models
