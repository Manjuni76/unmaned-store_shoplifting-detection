"""
Anomaly Score Attention 기반 분류기 학습 스크립트
STG-NF의 Feature + Anomaly Score를 결합하여 학습합니다.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

# 설정 import
from args import Config
from dataset import ShopliftingDataset
from anomaly_score_attention_classifier import (
    create_anomaly_score_attention_classifier, 
    AnomalyScoreAttentionClassifier
)
from utils_train import set_seed, print_section, print_model_info, save_checkpoint, load_checkpoint

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def load_stgnf_models(device):
    """부위별 STG-NF 모델 로드"""
    from model_builder import create_stgnf_model
    
    print_section("부위별 STG-NF 모델 로드")
    
    stg_nf_models = {}
    
    pbar = tqdm(Config.Joint.BODY_PARTS, desc="Loading STG-NF models", unit="model")
    for part_name in pbar:
        pbar.set_postfix({'Current': part_name})
        
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        # 'all' 부위는 subset이 None (전체 관절 사용)
        if joint_subset is None:
            num_joints = 18  # COCO 전체 관절 수
        else:
            num_joints = len(joint_subset)
        
        checkpoint_path = os.path.join(
            Config.Path.CHECKPOINT_DIR,
            f"stgnf_{part_name}.pth"
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"체크포인트 없음: {checkpoint_path}")
        
        model = create_stgnf_model(
            num_joints=num_joints,
            in_channels=Config.STGNF.IN_CHANNELS,
            num_frames=Config.Data.SEG_LEN,
            hidden_dim=Config.STGNF.HIDDEN_DIM,
            num_layers=Config.STGNF.NUM_LAYERS,
            subset_idx=joint_subset,
            device=device
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        stg_nf_models[part_name] = model
        print(f"\n [{part_name}] 로드 완료: {checkpoint_path}")
    
    print(f"\n총 {len(stg_nf_models)}개 부위 모델 로드 완료\n")
    return stg_nf_models


def get_sample_data(device):
    """샘플 데이터 생성 (Feature 차원 계산용)"""
    print("[INFO] 샘플 데이터 생성 중...")
    
    sample_data_dict = {}
    for part_name in Config.Joint.BODY_PARTS:
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        # 'all' 부위는 joint_subset이 None
        if joint_subset is None:
            num_joints = 18
        else:
            num_joints = len(joint_subset)
        
        # (B=1, C=3, T=SEG_LEN, V=num_joints)
        sample_data = torch.randn(
            1, 
            Config.STGNF.IN_CHANNELS, 
            Config.Data.SEG_LEN, 
            num_joints
        ).to(device)
        
        sample_data_dict[part_name] = sample_data
        print(f"  [{part_name}] shape: {sample_data.shape}")
    
    return sample_data_dict


def train_epoch(model, dataloader_dict, optimizer, criterion, device, epoch, total_epochs):
    """한 에포크 학습"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    total_batches = len(dataloader_dict[Config.Joint.BODY_PARTS[0]])
    
    batch_iter = tqdm(
        zip(*[dataloader_dict[part] for part in Config.Joint.BODY_PARTS]),
        total=total_batches,
        desc=f"Epoch {epoch+1}/{total_epochs}",
        unit="batch",
        ncols=120
    )
    
    for batch_idx, batch_data in enumerate(batch_iter):
        x_dict = {}
        labels = None
        
        for i, part in enumerate(Config.Joint.BODY_PARTS):
            data, label = batch_data[i]
            x_dict[part] = data.to(device)
            if labels is None:
                labels = label.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(x_dict)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.Attention.GRAD_CLIP)
        
        optimizer.step()
        
        # 통계
        total_loss += loss.item()
        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1
        
        # 진행상황 업데이트 (매 10 배치마다)
        if (batch_idx + 1) % 10 == 0:
            current_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
            batch_iter.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_loss/num_batches:.4f}',
                'Acc': f'{current_acc:.1f}%'
            })
    
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 클래스별 정확도
    normal_mask = np.array(all_labels) == 0
    abnormal_mask = np.array(all_labels) == 1
    
    normal_acc = accuracy_score(
        np.array(all_labels)[normal_mask], 
        np.array(all_preds)[normal_mask]
    ) if normal_mask.sum() > 0 else 0.0
    
    abnormal_acc = accuracy_score(
        np.array(all_labels)[abnormal_mask], 
        np.array(all_preds)[abnormal_mask]
    ) if abnormal_mask.sum() > 0 else 0.0
    
    return avg_loss, accuracy, normal_acc, abnormal_acc


def main():
    print("\n" + "="*80)
    print("Anomaly Score Attention 분류기 학습")
    print("="*80 + "\n")
    
    # 설정 출력
    print_section("학습 설정")
    print(f"  Device: {Config.Train.DEVICE}")
    print(f"  Batch Size: {Config.Attention.BATCH_SIZE}")
    print(f"  Learning Rate: {Config.Attention.LEARNING_RATE}")
    print(f"  Epochs: {Config.Attention.EPOCHS}")
    print(f"  Weight Decay: {Config.Attention.WEIGHT_DECAY}")
    print(f"  Gradient Clip: {Config.Attention.GRAD_CLIP}")
    print(f"  Early Stop Patience: {Config.Attention.EARLY_STOP_PATIENCE}")
    print(f"\n  Embedding Dim: {Config.Attention.EMBED_DIM}")
    print(f"  Score Embed Dim: {Config.Attention.SCORE_EMBED_DIM}")
    print(f"  Num Heads: {Config.Attention.NUM_HEADS}")
    print(f"  Num Encoder Layers: {Config.Attention.NUM_ENCODER_LAYERS}")
    print(f"  Dropout: {Config.Attention.DROPOUT}")
    print(f"\n  Focal Loss Gamma: {Config.Attention.FOCAL_GAMMA}")
    print(f"  Normal Weight: {Config.Attention.NORMAL_WEIGHT}")
    print(f"  Abnormal Weight: {Config.Attention.ABNORMAL_WEIGHT}")
    
    # Seed 설정
    set_seed(Config.Train.SEED)
    
    # Device 설정
    device = torch.device(Config.Train.DEVICE)
    print(f"\n[INFO] Using device: {device}")
    
    # 1. STG-NF 모델 로드
    stg_nf_models = load_stgnf_models(device)
    
    # 2. 샘플 데이터 생성
    sample_data_dict = get_sample_data(device)
    
    # 3. Anomaly Score Attention 모델 생성
    print_section("Anomaly Score Attention 모델 생성")
    model = create_anomaly_score_attention_classifier(
        stg_nf_models_dict=stg_nf_models,
        sample_data_dict=sample_data_dict,
        num_classes=Config.Attention.NUM_CLASSES,
        embed_dim=Config.Attention.EMBED_DIM,
        score_embed_dim=Config.Attention.SCORE_EMBED_DIM,
        num_heads=Config.Attention.NUM_HEADS,
        num_encoder_layers=Config.Attention.NUM_ENCODER_LAYERS,
        dropout=Config.Attention.DROPOUT,
        device=device
    )
    
    print_model_info(model)
    
    # 4. 데이터셋 로드
    print_section("데이터셋 로드")
    
    datasets_dict = {}
    for part_name in Config.Joint.BODY_PARTS:
        dataset = ShopliftingDataset(
            json_path=Config.Path.MLP_TRAIN_JSON,
            skeleton_base_path=Config.Path.MLP_TRAIN_DATA_DIR,
            seg_len=Config.Data.SEG_LEN,
            seg_stride=Config.Data.MLP_TRAIN_STRIDE,
            joint_subset=Config.Joint.JOINT_SUBSET_MAP[part_name],
            normalize=Config.Data.NORMALIZE,
            apply_augmentation=False,
            vid_res=Config.Data.VID_RES,
            use_cache=True,
            load_per_batch=False,
            preprocess_cache=True
        )
        datasets_dict[part_name] = dataset
        print(f"  [{part_name}] samples: {len(dataset)}")
    
    # 클래스 분포 확인
    dataset_sample = datasets_dict[Config.Joint.BODY_PARTS[0]]
    labels = [dataset_sample[i][1] for i in range(len(dataset_sample))]
    class_counts = np.bincount(labels)
    
    print(f"\n[INFO] Class Distribution:")
    print(f"  Normal (0): {class_counts[0]} ({class_counts[0]/len(labels)*100:.2f}%)")
    print(f"  Abnormal (1): {class_counts[1]} ({class_counts[1]/len(labels)*100:.2f}%)")
    
    # --- [수정 시작] ---
    
    # 클래스 불균형을 해결하기 위한 WeightedRandomSampler 생성
    # 1. 각 샘플에 대한 가중치 계산 (희귀 클래스일수록 높은 가중치)
    labels_array = np.array(labels)
    class_weights = 1. / class_counts
    
    # 2. 각 샘플(인덱스)이 어떤 가중치를 갖는지 매핑
    # 예: [0.0001, 0.0001, 0.01, 0.0001, 0.01, ...]
    samples_weight = np.array([class_weights[label] for label in labels_array])
    samples_weight = torch.from_numpy(samples_weight).double()

    # 3. 샘플러 생성
    sampler = WeightedRandomSampler(
        weights=samples_weight, 
        num_samples=len(samples_weight), 
        replacement=True
    )
    print("\n[INFO] WeightedRandomSampler 적용. (클래스 불균형 해소)")

    # DataLoader 생성 (Sampler 사용)
    dataloaders_dict = {}
    for part_name in Config.Joint.BODY_PARTS:
        dataloader = DataLoader(
            datasets_dict[part_name],
            batch_size=Config.Attention.BATCH_SIZE,
            shuffle=False,  # Sampler 사용 시 shuffle=False로 설정해야 함
            sampler=sampler, # 생성한 샘플러 적용
            num_workers=Config.Train.NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=Config.Train.PREFETCH_FACTOR,
            persistent_workers=Config.Train.PERSISTENT_WORKERS
        )
        dataloaders_dict[part_name] = dataloader
    
    # 5. Optimizer & Loss
    print_section("Optimizer & Loss")
    
    # Optimizer (Attention Classifier만 학습)
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=Config.Attention.LEARNING_RATE,
        weight_decay=Config.Attention.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # 일반 CrossEntropyLoss (클래스 불균형 처리 OFF)
    criterion = nn.CrossEntropyLoss()
    print(f"[INFO] Loss: CrossEntropyLoss (no class weights)")
    
    # LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    print(f"[INFO] LR Scheduler: ReduceLROnPlateau")
    
    # 6. 학습 시작
    print_section("학습 시작")
    
    checkpoint_path = os.path.join(
        Config.Path.CHECKPOINT_DIR, 
        "anomaly_attention_classifier_best.pth"
    )
    last_checkpoint_path = os.path.join(
        Config.Path.CHECKPOINT_DIR, 
        "anomaly_attention_classifier_last.pth"
    )
    
    best_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(Config.Attention.EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{Config.Attention.EPOCHS}]")
        print(f"{'='*80}")
        
        # 학습
        train_loss, train_acc, normal_acc, abnormal_acc = train_epoch(
            model, dataloaders_dict, optimizer, criterion, device, epoch, Config.Attention.EPOCHS
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n[Epoch {epoch+1}/{Config.Attention.EPOCHS} Results]")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc*100:.2f}%")
        print(f"  Normal Acc: {normal_acc*100:.2f}%")
        print(f"  Abnormal Acc: {abnormal_acc*100:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # LR Scheduler
        scheduler.step(train_acc)
        
        # Best 모델 저장
        if train_acc > best_acc:
            best_acc = train_acc
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                path=checkpoint_path
            )
            print(f"  ✓ Best model 저장 (Acc: {train_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{Config.Attention.EARLY_STOP_PATIENCE}")
        
        # 매 에포크마다 last 체크포인트 저장 (이어서 학습용)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=train_loss,
            path=last_checkpoint_path
        )
        
        # Early Stopping
        if patience_counter >= Config.Attention.EARLY_STOP_PATIENCE:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n학습 완료!")
    print(f"  Best Accuracy: {best_acc*100:.2f}%")
    print(f"  Total Time: {total_time/60:.1f}분")
    
    print(f"\n최종 모델 저장 위치: {checkpoint_path}")


if __name__ == "__main__":
    main()
