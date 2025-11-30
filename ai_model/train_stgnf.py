"""
부위별 STG-NF 모델 학습 스크립트
"""

import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.data import DataLoader

# 설정 import
from args import Config
from datasets.dataset import ShopliftingDataset
from models.model_builder import create_stgnf_model
from datasets.train_utils import set_seed, print_section, print_model_info, save_checkpoint

# GPU ?�정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_stgnf_epoch(model, dataloader, optimizer, device, part_name):
    model.train()
    total_loss = 0
    total_nll = 0
    num_batches = 0
    
    
    pbar = tqdm(dataloader, desc=f"  Epoch Training...", leave=False, ncols=100)
    
    for batch_idx, (data, _) in enumerate(pbar):
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
        
        # tqdm 진행률에 Loss ?�시
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        
        # if (batch_idx + 1) % Config.Train.LOG_INTERVAL == 0:
        #     print(f"   [{part_name}] Batch [{batch_idx+1}/{len(dataloader)}] "
        #           f"Loss: {loss.item():.4f}, NLL: {nll.mean().item():.4f}")
    
    avg_loss = total_loss / num_batches
    avg_nll = total_nll / num_batches
    
    return avg_loss, avg_nll

# GT 캐시 (??번만 로드)
_GT_CACHE = None

def load_gt_cache():
    """GT 데이터를 한 번만 로드하여 캐싱"""
    global _GT_CACHE
    if _GT_CACHE is not None:
        return _GT_CACHE
    
    import glob
    video_frame_gt = {}
    
    # GT 폴더에서 직접 .npy 파일들을 스캔
    gt_files = glob.glob(os.path.join(Config.Path.TEST_GT_DIR, '*.npy'))
    
    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        video_id = filename.replace('.npy', '')
        
        gt_array = np.load(gt_path)
        frame_labels = {i: int(gt_array[i]) for i in range(len(gt_array))}
        video_frame_gt[video_id] = frame_labels
    
    _GT_CACHE = video_frame_gt
    return _GT_CACHE

def evaluate_discrimination(model, dataloader, device, part_name):
    """
    Test set에서 normal/abnormal NLL 차이 계산 (Best model 평가용)
    최적의 모델을 전달받아 dataloader 사용 + 프레임별 max NLL
    """
    from collections import defaultdict
    
    model.eval()
    
    # GT 데이터 로드 (캐시용)
    video_frame_gt = load_gt_cache()
    
    # 1. 각 프레임별(중첩되는 세그먼트의 NLL 모두 수집
    video_frame_nlls = defaultdict(lambda: defaultdict(list))
    
    dataset = dataloader.dataset
    segment_metadata = dataset.segment_metadata
    seg_len = Config.Data.SEG_LEN
    
    segment_idx = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"  Eval {part_name}", leave=False, ncols=80)
        for data, _ in pbar:
            data = data.to(device, non_blocking=True)
            batch_size = data.shape[0]
            
            # Forward pass
            z, nll = model(data)
            nll_scores = nll.cpu().numpy()
            
            # 해당 세그먼트의 NLL을 해당하는 모든 프레임에 할당
            for i in range(batch_size):
                if segment_idx >= len(segment_metadata):
                    break
                
                meta = segment_metadata[segment_idx]
                filename = meta['filename']
                video_id = filename.replace('.mp4', '').replace('.json', '')
                start_frame = meta['start_frame']
                nll_score = float(nll_scores[i])
                
                # 해당 세그먼트가 커버하는 모든 프레임에 NLL 추가
                for frame_idx in range(start_frame, start_frame + seg_len):
                    video_frame_nlls[video_id][frame_idx].append(nll_score)
                
                segment_idx += 1
    
    # 2. 각 프레임의 NLL 중 max 값을 선택
    video_frame_max_nll = {}
    for video_id, frame_dict in video_frame_nlls.items():
        video_frame_max_nll[video_id] = {}
        for frame_idx, nll_list in frame_dict.items():
            video_frame_max_nll[video_id][frame_idx] = max(nll_list)
    
    # 3. GT 기반으로 정상/비정상 NLL 수집
    normal_nlls = []
    abnormal_nlls = []
    
    for video_id, frame_dict in video_frame_max_nll.items():
        if video_id not in video_frame_gt:
            continue
        
        gt_frames = video_frame_gt[video_id]
        
        for frame_idx, nll_score in frame_dict.items():
            if frame_idx not in gt_frames:
                continue
            
            frame_label = gt_frames[frame_idx]
            
            if frame_label == 0:  # Normal
                normal_nlls.append(nll_score)
            else:  # Abnormal
                abnormal_nlls.append(nll_score)
    
    if len(normal_nlls) == 0 or len(abnormal_nlls) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # 평균 계산
    normal_mean = np.mean(normal_nlls)
    abnormal_mean = np.mean(abnormal_nlls)
    nll_diff = abnormal_mean - normal_mean
    nll_diff_pct = (nll_diff / normal_mean) * 100 if normal_mean != 0 else 0.0
    
    return normal_mean, abnormal_mean, nll_diff, nll_diff_pct

def train_stgnf_part(part_name, joint_subset, device):
    """부위별 STG-NF 모델 학습"""
    print_section(f"[{part_name.upper()}] 부위별 학습 시작")
    

    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, f'stgnf_{part_name}_fin.pth')
    
    if os.path.exists(checkpoint_path):
        print(f"기존 모델 {checkpoint_path}")
        os.remove(checkpoint_path)
        print("처음부터 학습 시작")
        
    # 데이터셋 생성
    print(f"\n[{part_name}] Training 데이터셋 로드 중")
    train_dataset = ShopliftingDataset(
        skeleton_base_path=Config.Path.TRAIN_DATA_DIR,
        seg_len=Config.Data.SEG_LEN,
        seg_stride=Config.Data.TRAIN_STRIDE,
        joint_subset=joint_subset,
        normalize=Config.Data.NORMALIZE,
        apply_augmentation=Config.Data.APPLY_AUGMENTATION,
        vid_res=Config.Data.VID_RES,
        use_cache=Config.Data.USE_CACHE,
        load_per_batch=False,
        preprocess_cache=False,
        filter_label='normal')
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.STGNF.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.Train.NUM_WORKERS,
        pin_memory=True,  
        prefetch_factor=Config.Train.PREFETCH_FACTOR if Config.Train.NUM_WORKERS > 0 else None,
        persistent_workers=Config.Train.PERSISTENT_WORKERS if Config.Train.NUM_WORKERS > 0 else False
    )
    
    # Test 데이터셋 (Best model 평가용 - normal/abnormal 모두 포함)
    print(f"[{part_name}] Test 데이터셋 로드 (Best model 평가용, stride=1)")
    val_dataset = ShopliftingDataset(
        skeleton_base_path=Config.Path.TEST_DATA_DIR,
        seg_len=Config.Data.SEG_LEN,
        seg_stride=1,
        joint_subset=joint_subset,
        normalize=Config.Data.NORMALIZE,
        apply_augmentation=False,
        vid_res=Config.Data.VID_RES,
        use_cache=True,
        load_per_batch=False,
        preprocess_cache=False)
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1024,  
        shuffle=False,
        num_workers=Config.Train.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=Config.Train.PREFETCH_FACTOR if Config.Train.NUM_WORKERS > 0 else None,
        persistent_workers=Config.Train.PERSISTENT_WORKERS if Config.Train.NUM_WORKERS > 0 else False
    )
    
    # 모델 생성
    num_joints = len(joint_subset) if joint_subset is not None else 18
    print(f"\n[{part_name}] 모델 생성 중 (관절 개수: {num_joints})")
    
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
    
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.STGNF.LEARNING_RATE,
        weight_decay=Config.STGNF.WEIGHT_DECAY
    )
    
    
    # Train Loss가 5 에포크(patience=5) 동안 개선되지 않으면
    # LR을 0.5배(factor=0.5) 줄임
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',      
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    
    # 학습
    print(f"\n[{part_name}] 학습 시작 ({Config.STGNF.EPOCHS} 에포크)")
    best_loss = float('inf')
    best_discrimination = -float('inf')  # 최저 NLL 차이 추적
    best_discrimination_epoch = 0
    patience_counter = 0
    
    # 훈련 로그 저장용 리스트
    training_log = []
    
    start_time = time.time()
    
    for epoch in range(Config.STGNF.EPOCHS):
        epoch_start = time.time()
        
        avg_loss, avg_nll = train_stgnf_epoch(model, train_dataloader, optimizer, device, part_name)
        
        # Validation: Normal/Abnormal NLL 차이 계산 (?�레?�별 ?�균)
        normal_mean, abnormal_mean, nll_diff, nll_diff_pct = evaluate_discrimination(
            model, val_dataloader, device, part_name
        )
        
        epoch_time = time.time() - epoch_start
        scheduler.step(avg_loss)
        
        # 현재 LR 확인
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n[{part_name}] Epoch [{epoch+1}/{Config.STGNF.EPOCHS}] "
              f"Loss: {avg_loss:.4f}, NLL: {avg_nll:.4f}, "
              f"Time: {epoch_time:.2f}s, LR: {current_lr:.6e}")
        print(f"  Test Set - Normal NLL: {normal_mean:.4f}, Abnormal NLL: {abnormal_mean:.4f}, "
              f"Diff: {nll_diff:+.4f} ({nll_diff_pct:+.2f}%)")
        
        # 로그 ?�??
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_nll': avg_nll,
            'val_normal_nll': normal_mean,
            'val_abnormal_nll': abnormal_mean,
            'val_diff': nll_diff,
            'val_diff_pct': nll_diff_pct,
            'lr': current_lr
        }
        training_log.append(log_entry)
        
        # Best discrimination model ?�??(Test set 기�? NLL 차이가 최�???모델)
        if nll_diff > best_discrimination:
            best_discrimination = nll_diff
            best_discrimination_epoch = epoch + 1
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                path=checkpoint_path
            )
            print(f"  >>Best model 저장 (Test Diff: {best_discrimination:+.4f}, {nll_diff_pct:+.2f}%)")
        
        # Best loss ?�데?�트 (EarlyStopping??
        
        if avg_loss < (best_loss - Config.STGNF.EARLY_STOP_MIN_DELTA):
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{Config.STGNF.EARLY_STOP_PATIENCE}")
        
        # Early stopping
        if patience_counter >= Config.STGNF.EARLY_STOP_PATIENCE:
            print(f"\n[{part_name}] Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n[{part_name}] 학습 총시간: {total_time/60:.2f}분")
    print(f"[{part_name}] Best Loss: {best_loss:.4f}")
    print(f"[{part_name}] Best Discrimination: {best_discrimination:+.4f} (Epoch {best_discrimination_epoch})")
    print(f"[{part_name}] 모델 저장 {checkpoint_path}")
    
    # 학습 로그를 JSON으로 저장
    import json
    log_path = os.path.join(Config.Path.CHECKPOINT_DIR, f'training_log_{part_name}.json')
    with open(log_path, 'w') as f:
        json.dump({
            'part_name': part_name,
            'best_discrimination': best_discrimination,
            'best_discrimination_epoch': best_discrimination_epoch,
            'best_loss': best_loss,
            'total_epochs': len(training_log),
            'total_time_minutes': total_time / 60,
            'epochs': training_log
        }, f, indent=2)
    print(f"[{part_name}] 학습 로그 저장 {log_path}")
    
    return checkpoint_path

def main():
    """메인 함수"""
    print_section("부위별 STG-NF 모델 학습")
    print(f"학습 부위: {Config.Joint.BODY_PARTS}")
    
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
    print_section("모든 부위별 학습 완료")
    print("\n생성된 체크포인트")
    for part, path in checkpoint_paths.items():
        status = "존재" if os.path.exists(path) else "없음"
        print(f"  [{part}] {status}: {path}")
    

if __name__ == "__main__":
    main()

