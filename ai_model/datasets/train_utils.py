"""
유틸리티 함수들
시드 설정, 로깅 등
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """모든 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"랜덤 시드 고정: {seed}")


def print_section(title, width=80):
    """섹션 제목 출력"""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection(title, width=80):
    """서브섹션 제목 출력"""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def count_parameters(model):
    """모델의 학습 가능한 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def print_model_info(model, name="Model"):
    """모델 정보 출력"""
    params = count_parameters(model)
    print(f"\n{name} 정보:")
    print(f"  총 파라미터: {params['total']:,}")
    print(f"  학습 가능: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")


def format_time(seconds):
    """시간을 읽기 쉬운 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_checkpoint(model, optimizer, epoch, loss, path):
    """체크포인트 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"체크포인트 저장됨: {path}")


def load_checkpoint(model, optimizer, path, device='cuda'):
    """체크포인트 로드"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    print(f"체크포인트 로드됨: {path} (Epoch {epoch}, Loss {loss:.4f})")
    return epoch, loss
