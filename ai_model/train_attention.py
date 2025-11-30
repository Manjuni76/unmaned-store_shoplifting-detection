"""Attention 분류기 학습 스크립트 (검증 속도 최적화 버전)

기능:
 - STG-NF 모델을 Frozen Feature Extractor로 사용
 - Attention Classifier 학습
 - [최적화] DataLoader를 활용한 고속 프레임 단위 평가 (CPU 병목 제거)
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from args import Config
from datasets.dataset_folder_scan import FolderScanDataset
from models.stgnf_loader import load_all_stgnf_models
from models.model_builder import create_attention_classifier

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha: 클래스 불균형 처리를 위한 가중치 (여기선 None 또는 리스트)
        self.alpha = alpha

    def forward(self, inputs, targets):
        # inputs: (B, C) - Logits
        # targets: (B) - Labels
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt: 정답 클래스에 대한 확률
        
        # Focal Loss 수식: (1 - pt)^gamma * log(pt)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            # Alpha 적용 (클래스별 가중치)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
def train_one_epoch(model, dataloader, optimizer, device):
    """학습 루프 (세그먼트 단위)"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    if Config.Attention.USE_FOCAL_LOSS:
        # Focal Loss 사용
        criterion = FocalLoss(gamma=Config.Attention.FOCAL_GAMMA)
    else:
        # 기존 Cross Entropy (Label Smoothing 적용 추천)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="[Train]", leave=True, ncols=100)
    for data, labels in pbar:
        data = data.to(device)  # (B, C, T, V)
        labels = labels.to(device)

        optimizer.zero_grad()

        # x_dict: 부위별 subset 슬라이싱
        x_dict = {}
        for part in model.part_names:
            subset = Config.Joint.JOINT_SUBSET_MAP[part]
            if subset is not None:
                x_dict[part] = data[:, :, :, subset]
            else:
                x_dict[part] = data
        
        logits = model(x_dict)  # (B, num_classes)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/batch_size*100:.1f}%"})

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples * 100 if total_samples > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    return avg_loss, avg_acc, f1


@torch.no_grad()
def evaluate_frame_level(model, val_loader, device):
    """
    [최적화된 검증 함수]
    DataLoader를 직접 사용하여 GPU 배치를 처리하므로 속도가 획기적으로 빠릅니다.
    """
    model.eval()
    
    # 결과를 저장할 딕셔너리
    # key: video_id, value: {frame_idx: [score1, score2...]}
    video_frame_scores = {}
    
    # GT 및 메타데이터 접근을 위해 dataset 참조
    val_dataset = val_loader.dataset
    seg_len = Config.Data.SEG_LEN
    
    # Val Loader 순회 (이미 전처리된 배치를 받음)
    # 중요: 검증 시 shuffle=False여야 순서대로 메타데이터 매칭 가능
    pbar = tqdm(val_loader, desc="[Valid-Inference]", leave=False, ncols=100)
    
    global_idx = 0 # 전체 데이터셋에서의 인덱스 추적용

    for data, _ in pbar: # 검증 라벨은 사용 안 함 (GT 파일 별도 로드)
        data = data.to(device)
        batch_size = data.size(0)
        
        # 1. 입력 준비 (부위별 슬라이싱)
        x_dict = {}
        for part in model.part_names:
            subset = Config.Joint.JOINT_SUBSET_MAP[part]
            if subset is not None:
                x_dict[part] = data[:, :, :, subset]
            else:
                x_dict[part] = data
        
        # 2. 모델 예측 (GPU 일괄 처리)
        logits = model(x_dict)
        # Class 1(도난)일 확률
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy() 
        
        # 3. 점수 매핑 (CPU 연산)
        # 배치 내 각 샘플을 원래 비디오/프레임 위치에 할당
        for i in range(batch_size):
            # 현재 처리 중인 세그먼트의 메타데이터 가져오기
            # shuffle=False이므로 global_idx로 순차 접근 가능
            meta = val_dataset.segment_metadata[global_idx + i]
            
            video_id = meta['filename']
            start_frame = meta['start_frame']
            score = float(probs[i])
            
            if video_id not in video_frame_scores:
                video_frame_scores[video_id] = {}
            
            # 해당 세그먼트가 커버하는 프레임들에 점수 할당
            # 딕셔너리 조회 최적화
            v_dict = video_frame_scores[video_id]
            for offset in range(seg_len):
                frame_idx = start_frame + offset
                if frame_idx not in v_dict:
                    v_dict[frame_idx] = []
                v_dict[frame_idx].append(score)
        
        global_idx += batch_size

    # 4. 결과 집계 (Max Aggregation) 및 GT 비교
    y_true = []
    y_scores = []
    
    # print("  Aggregating scores...") # 진행바와 겹치면 주석 처리
    for vid, frames in video_frame_scores.items():
        # GT Map에 없는 비디오는 스킵
        if vid not in val_dataset.gt_map:
            continue
            
        try:
            gt_array = np.load(val_dataset.gt_map[vid])
        except:
            continue
            
        for f, scores in frames.items():
            # 프레임 범위 체크
            if f >= len(gt_array): continue
            
            # 겹치는 세그먼트 중 가장 높은 도난 확률을 해당 프레임 점수로 사용
            final_score = np.max(scores)
            
            y_true.append(int(gt_array[f]))
            y_scores.append(final_score)

    # 5. F1 Score 계산 (Threshold 탐색)
    if not y_true:
        return 0.0, 0.0, 0.0, 0.5

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    best_f1 = 0.0
    best_th = 0.5
    
    # 0.1 ~ 0.99 사이에서 최적의 임계값 탐색
    thresholds = np.linspace(0.1, 0.99, 50)
    
    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    # 최종 Acc 계산
    y_pred_best = (y_scores >= best_th).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    acc = (cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0

    return 0.0, acc * 100.0, best_f1 * 100.0, best_th


def main():
    print("STG-NF 잠재벡터 기반 Attention 분류 학습 시작")
    device = torch.device(Config.Train.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    Config.Path.create_dirs()

    # 1. STG-NF 모델 로드
    print("\nStep 1: STG-NF 부위별 모델 로드")
    stgnf_models = load_all_stgnf_models(device)

    # 2. Dataset 생성
    print("\nStep 2: Dataset 로드 및 메모리 캐싱")
    
    # Train Dataset
    train_dataset = FolderScanDataset(
        skeleton_dir=Config.Path.ATTENTION_TRAIN_DATA_DIR,
        gt_dir=Config.Path.ATTENTION_TRAIN_GT_DIR,
        seg_len=Config.Data.SEG_LEN,
        seg_stride=Config.Data.TRAIN_STRIDE,
        joint_subset=None, 
        normalize=Config.Data.NORMALIZE,
        apply_augmentation=Config.Data.APPLY_AUGMENTATION,
        vid_res=Config.Data.VID_RES,
        use_cache=True,
        preprocess_cache=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.Attention.BATCH_SIZE,
        shuffle=True,
        num_workers=0, # 시스템에 맞게 조절
        pin_memory=True,
        #persistent_workers=True
    )

    # Val Dataset
    val_dataset = FolderScanDataset(
        skeleton_dir=Config.Path.TEST_DATA_DIR,
        gt_dir=Config.Path.TEST_GT_DIR,
        seg_len=Config.Data.SEG_LEN,
        seg_stride=1, # 검증은 stride 1로 촘촘하게
        joint_subset=None,
        normalize=Config.Data.NORMALIZE,
        apply_augmentation=False,
        vid_res=Config.Data.VID_RES,
        use_cache=True,
        preprocess_cache=True
    )
    # [중요] 검증용 Loader: shuffle=False 필수!
    val_loader = DataLoader(
        val_dataset,
        batch_size=512, # 검증은 배치를 키워도 됨
        shuffle=False, 
        num_workers=0,
        pin_memory=True,
        #persistent_workers=True
    )

    # 3. 샘플 입력 준비
    print("\nStep 3: Feature 차원 계산 샘플 준비")
    sample_batch = next(iter(train_loader))[0][:1]
    sample_data_dict = {}
    for part in stgnf_models.keys():
        subset = Config.Joint.JOINT_SUBSET_MAP[part]
        if subset is not None:
            sample_data_dict[part] = sample_batch[:, :, :, subset].to(device)
        else:
            sample_data_dict[part] = sample_batch.to(device)

    # 4. Attention 모델 생성
    print("\nStep 4: Attention 모델 생성")
    attention_model = create_attention_classifier(
        stg_nf_models_dict=stgnf_models,
        sample_data_dict=sample_data_dict,
        num_classes=Config.Attention.NUM_CLASSES,
        embed_dim=Config.Attention.EMBED_DIM,
        num_heads=Config.Attention.NUM_HEADS,
        num_encoder_layers=Config.Attention.NUM_ENCODER_LAYERS,
        dropout=Config.Attention.DROPOUT,
        device=str(device)
    )

    # 5. Optimizer
    print("\nStep 5: Optimizer 설정")
    params = [p for p in attention_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=Config.Attention.LEARNING_RATE, weight_decay=Config.Attention.WEIGHT_DECAY)

    # 6. 학습 루프
    epochs = Config.Attention.EPOCHS
    best_val_f1 = 0.0
    log = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(attention_model, train_loader, optimizer, device)
        
        # Validation (수정됨: Loader 전달)
        val_loss, val_acc, val_f1, val_best_th = evaluate_frame_level(attention_model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        print(f"\n결과: Epoch {epoch} - {epoch_time:.1f}초")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%")
        print(f"  Val(Frame) Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}% (best_th={val_best_th:.2f})")

        log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_frame_f1': val_f1,
            'val_best_th': val_best_th
        })

        # 베스트 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(Config.Path.CHECKPOINT_DIR, 'attention_best.pth')
            torch.save({
                'model_state_dict': attention_model.state_dict(),
                'val_frame_f1': val_f1,
                'epoch': epoch
            }, save_path)
            print(f"  ✓ Best 업데이트 저장 -> {save_path}")

    total_min = (time.time() - start_time) / 60
    print(f"\n학습 완료! 총 시간: {total_min:.2f}분, Best Val Frame F1: {best_val_f1:.2f}%")

    # 로그 저장
    log_path = os.path.join(Config.Path.CHECKPOINT_DIR, 'attention_training_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({'best_val_f1': best_val_f1, 'epochs': log}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()