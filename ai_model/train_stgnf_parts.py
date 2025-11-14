"""
부위별 STG-NF 모델 학습 스크립트
정상 데이터만 사용하여 각 부위(head, arms, body, legs, all)별로 STG-NF 모델을 학습하고 저장합니다.
"""

import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 설정 import
from args import Config
from dataset import ShopliftingDataset
from model_builder import create_stgnf_model
from utils_train import set_seed, print_section, print_model_info, save_checkpoint

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_stgnf_epoch(model, dataloader, optimizer, device, part_name):
    """STG-NF 한 에포크 학습"""
    model.train()
    total_loss = 0
    total_nll = 0
    num_batches = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)  # (B, C=2, T, V)
        
        optimizer.zero_grad()
        
        # Forward pass
        z, nll = model(data)
        loss = nll.mean()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.STGNF.GRAD_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        total_nll += nll.mean().item()
        num_batches += 1
        
        if (batch_idx + 1) % Config.Train.LOG_INTERVAL == 0:
            print(f"  [{part_name}] Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}, NLL: {nll.mean().item():.4f}")
    
    avg_loss = total_loss / num_batches
    avg_nll = total_nll / num_batches
    
    return avg_loss, avg_nll


def train_stgnf_part(part_name, joint_subset, device):
    """부위별 STG-NF 모델 학습"""
    print_section(f"[{part_name.upper()}] 부위 학습 시작")
    
    # 체크포인트 경로
    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, f'stgnf_{part_name}.pth')
    
    # 이미 학습된 모델이 있으면 스킵
    if os.path.exists(checkpoint_path):
        print(f"[INFO] 이미 학습된 모델 존재: {checkpoint_path}")
        print(f"[INFO] {part_name} 부위 학습 스킵")
        return checkpoint_path
    
    # 데이터셋 생성
    print(f"\n[{part_name}] 데이터셋 로드 중...")
    dataset = ShopliftingDataset(
        json_path=Config.Path.TRAIN_JSON,
        skeleton_base_path=Config.Path.TRAIN_DATA_DIR,
        seg_len=Config.Data.SEG_LEN,
        seg_stride=Config.Data.TRAIN_STRIDE,
        joint_subset=joint_subset,
        normalize=Config.Data.NORMALIZE,
        apply_augmentation=Config.Data.APPLY_AUGMENTATION,
        vid_res=Config.Data.VID_RES,
        use_cache=Config.Data.USE_CACHE,  # 스켈레톤 데이터 캐싱
        load_per_batch=False,  # 사전에 모든 세그먼트 생성
        preprocess_cache=True  # 전처리된 데이터 캐싱 (학습 속도 대폭 향상)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=Config.STGNF.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.Train.NUM_WORKERS,
        pin_memory=Config.Train.PIN_MEMORY,
        prefetch_factor=Config.Train.PREFETCH_FACTOR if Config.Train.NUM_WORKERS > 0 else None,
        persistent_workers=Config.Train.PERSISTENT_WORKERS if Config.Train.NUM_WORKERS > 0 else False
    )
    
    # 모델 생성
    num_joints = len(joint_subset) if joint_subset is not None else 18
    print(f"\n[{part_name}] 모델 생성 중... (관절 수: {num_joints})")
    
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
    
    print_model_info(model, f"{part_name.upper()} STG-NF")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.STGNF.LEARNING_RATE,
        weight_decay=Config.STGNF.WEIGHT_DECAY
    )
    
    # 학습
    print(f"\n[{part_name}] 학습 시작 ({Config.STGNF.EPOCHS} 에포크)")
    best_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(Config.STGNF.EPOCHS):
        epoch_start = time.time()
        
        avg_loss, avg_nll = train_stgnf_epoch(model, dataloader, optimizer, device, part_name)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n[{part_name}] Epoch [{epoch+1}/{Config.STGNF.EPOCHS}] "
              f"Loss: {avg_loss:.4f}, NLL: {avg_nll:.4f}, Time: {epoch_time:.2f}s")
        
        # Best model 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                path=checkpoint_path
            )
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.STGNF.EARLY_STOP_PATIENCE:
            print(f"[{part_name}] Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n[{part_name}] 학습 완료! 총 시간: {total_time/60:.2f}분")
    print(f"[{part_name}] Best Loss: {best_loss:.4f}")
    print(f"[{part_name}] 모델 저장: {checkpoint_path}")
    
    return checkpoint_path


def main():
    """메인 함수"""
    print_section("부위별 STG-NF 모델 학습")
    print(f"학습할 부위: {Config.Joint.BODY_PARTS}")
    
    # 시드 설정
    set_seed(Config.Train.SEED)
    
    # 디바이스 설정
    device = torch.device(Config.Train.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # 체크포인트 디렉토리 생성
    Config.Path.create_dirs()
    
    # 부위별 학습
    checkpoint_paths = {}
    
    for part_name in Config.Joint.BODY_PARTS:
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        checkpoint_path = train_stgnf_part(
            part_name=part_name,
            joint_subset=joint_subset,
            device=device
        )
        
        checkpoint_paths[part_name] = checkpoint_path
        
        print("\n" + "-"*80 + "\n")
    
    # 최종 결과
    print_section("모든 부위 학습 완료!")
    print("\n저장된 체크포인트:")
    for part, path in checkpoint_paths.items():
        status = "✓ 존재" if os.path.exists(path) else "✗ 없음"
        print(f"  [{part}] {status}: {path}")
    
    print("\n다음 단계: python train_attention_classifier.py")


if __name__ == "__main__":
    main()
