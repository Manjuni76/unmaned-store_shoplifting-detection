"""
Attention 기반 분류기 학습 스크립트
사전 학습된 부위별 STG-NF 모델들을 로드하여 Attention 분류기만 학습합니다.
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
from attention_classifier import create_attention_classifier, PartAttentionClassifier
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


def save_checkpoint(model, optimizer, epoch, loss, accuracy, save_path):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, save_path)


def train_epoch(model, dataloader_dict, optimizer, criterion, device):
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
        desc="Training",
        unit="batch"
    )
    
    for batch_data in batch_iter:
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
        
        # 클래스별 예측 분포
        normal_count = (predicted == 0).sum().item()
        abnormal_count = (predicted == 1).sum().item()
        batch_iter.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/num_batches:.4f}',
            'N/Ab': f'{normal_count}/{abnormal_count}'
        })
    
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 클래스별 정확도
    normal_mask = np.array(all_labels) == 0
    abnormal_mask = np.array(all_labels) == 1
    normal_acc = accuracy_score(np.array(all_labels)[normal_mask], np.array(all_preds)[normal_mask]) if normal_mask.sum() > 0 else 0
    abnormal_acc = accuracy_score(np.array(all_labels)[abnormal_mask], np.array(all_preds)[abnormal_mask]) if abnormal_mask.sum() > 0 else 0
    
    print(f"\n[Train] Normal Acc: {normal_acc:.4f}, Abnormal Acc: {abnormal_acc:.4f}")
    
    return avg_loss, accuracy


def main():
    """메인 함수"""
    print_section("Attention 기반 분류기 학습")
    
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='Attention Classifier Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='체크포인트 경로 (이어서 학습)')
    args = parser.parse_args()
    
    # 시드 설정
    set_seed(Config.Train.SEED)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}\n")
    
    # 체크포인트 디렉토리 확인
    if not os.path.exists(Config.Path.CHECKPOINT_DIR):
        os.makedirs(Config.Path.CHECKPOINT_DIR)
    
    # 체크포인트 경로 설정
    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "attention_classifier_final.pth")
    last_checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "attention_classifier_last.pth")
    
    # 자동 resume: last 체크포인트가 있으면 사용
    if args.resume is None and os.path.exists(last_checkpoint_path):
        print(f"[INFO] 마지막 체크포인트 발견: {last_checkpoint_path}")
        response = input("이어서 학습하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            args.resume = last_checkpoint_path
    
    # 체크포인트 존재 확인
    if not args.resume and os.path.exists(checkpoint_path):
        print(f"[INFO] 체크포인트 발견: {checkpoint_path}")
        response = input("덮어쓰기 (y) / 이어서 학습 (r) / 학습 스킵 (n): ")
        if response.lower() == 'n':
            print("[INFO] Attention 분류기 학습 스킵")
            return
        elif response.lower() == 'r':
            args.resume = checkpoint_path
    
    # STG-NF 모델들 로드
    stg_nf_models = load_stgnf_models(device)
    
    # 샘플 데이터 생성
    sample_data_dict = get_sample_data(device)
    
    # Attention 분류기 생성
    print_section("Attention 분류기 생성")
    model = create_attention_classifier(
        stg_nf_models_dict=stg_nf_models,
        sample_data_dict=sample_data_dict,
        num_classes=Config.Attention.NUM_CLASSES,
        embed_dim=Config.Attention.EMBED_DIM,
        num_heads=Config.Attention.NUM_HEADS,
        num_encoder_layers=Config.Attention.NUM_ENCODER_LAYERS,
        dropout=Config.Attention.DROPOUT,
        device=device
    )
    
    print_model_info(model, "Attention Classifier")
    
    # 데이터셋 로드
    print_section("데이터셋 로드")
    
    train_datasets = {}
    train_loaders = {}
    
    print("[INFO] 데이터셋 생성 (부위별 joint 추출)...")
    
    for part_name in Config.Joint.BODY_PARTS:
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        dataset = ShopliftingDataset(
            json_path=Config.Path.MLP_TRAIN_JSON,
            skeleton_base_path=Config.Path.MLP_TRAIN_DATA_DIR,
            seg_len=Config.Data.SEG_LEN,
            seg_stride=Config.Data.MLP_TRAIN_STRIDE,
            joint_subset=joint_subset,
            normalize=Config.Data.NORMALIZE,
            apply_augmentation=True,
            vid_res=Config.Data.VID_RES,
            use_cache=True,          # 스켈레톤 JSON 메모리 캐싱
            load_per_batch=False,    # 모든 세그먼트 사전 생성
            preprocess_cache=False   # augmentation 사용시 전처리 캐시 비활성화
        )
        
        train_datasets[part_name] = dataset
        print(f"  [{part_name}] {len(dataset)} 샘플")
    
    # Oversampling 가중치 계산
    first_dataset = train_datasets[Config.Joint.BODY_PARTS[0]]
    labels = first_dataset.segment_labels
    normal_count = sum(1 for l in labels if l == 0)
    abnormal_count = sum(1 for l in labels if l == 1)
    
    print(f"\n[INFO] 데이터 분포:")
    print(f"  Normal: {normal_count:,}개 ({normal_count/len(labels)*100:.1f}%)")
    print(f"  Abnormal: {abnormal_count:,}개 ({abnormal_count/len(labels)*100:.1f}%)")
    
    # Oversampling
    sample_weights = [1.0 if l == 0 else normal_count / abnormal_count for l in labels]
    
    print(f"\n[INFO] Oversampling 적용:")
    print(f"  Normal weight: 1.0")
    print(f"  Abnormal weight: {normal_count/abnormal_count:.1f}x")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # DataLoader 생성
    for part_name in Config.Joint.BODY_PARTS:
        loader = DataLoader(
            train_datasets[part_name],
            batch_size=Config.Attention.BATCH_SIZE,
            sampler=sampler,
            num_workers=Config.Train.NUM_WORKERS,
            pin_memory=True
        )
        train_loaders[part_name] = loader
    
    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.Attention.LEARNING_RATE,
        weight_decay=Config.Attention.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Class Weights (from Config.Attention)
    normal_weight = Config.Attention.NORMAL_WEIGHT
    abnormal_weight = Config.Attention.ABNORMAL_WEIGHT
    
    class_weights = torch.tensor([normal_weight, abnormal_weight], dtype=torch.float32).to(device)
    
    print(f"\n[INFO] Class Weights:")
    print(f"  Normal: {class_weights[0]:.3f}")
    print(f"  Abnormal: {class_weights[1]:.3f}")
    
    # Focal Loss (from Config.Attention)
    criterion = FocalLoss(alpha=class_weights, gamma=Config.Attention.FOCAL_GAMMA)
    print(f"[INFO] Focal Loss (gamma={Config.Attention.FOCAL_GAMMA})")
    
    # LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    print(f"[INFO] LR Scheduler: ReduceLROnPlateau")
    
    # 체크포인트에서 이어서 학습
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        print(f"\n[INFO] 체크포인트 로드: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('accuracy', 0.0)
        print(f"[INFO] Epoch {start_epoch}부터 재시작 (Best Acc: {best_acc*100:.2f}%)")
    
    # 학습
    print_section(f"학습 시작 ({Config.Attention.EPOCHS} 에포크)")
    
    patience_counter = 0
    start_time = time.time()
    
    epoch_iter = tqdm(range(start_epoch, Config.Attention.EPOCHS), desc="Epochs", unit="epoch")
    
    for epoch in epoch_iter:
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loaders, optimizer, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        epoch_iter.set_postfix({
            'Loss': f'{train_loss:.4f}',
            'Acc': f'{train_acc*100:.1f}%',
            'Time': f'{epoch_time:.1f}s'
        })
        
        print(f"\nEpoch [{epoch+1}/{Config.Attention.EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # LR Scheduler
        scheduler.step(train_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current LR: {current_lr:.6f}")
        
        # Best model 저장
        if train_acc > best_acc:
            best_acc = train_acc
            patience_counter = 0
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                accuracy=train_acc,
                save_path=checkpoint_path
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
            accuracy=train_acc,
            save_path=last_checkpoint_path
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
