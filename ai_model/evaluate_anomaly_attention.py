"""
Anomaly Score Attention 기반 분류기 평가 스크립트
학습된 모델의 성능을 테스트 데이터로 평가합니다.
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 설정 import
from args import Config
from dataset import ShopliftingDataset
from anomaly_score_attention_classifier import create_anomaly_score_attention_classifier
from utils_train import set_seed, print_section

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def calculate_eer(y_true, y_scores):
    """EER(Equal Error Rate) 계산"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # FPR과 FNR이 같아지는 지점 찾기
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer_value = fpr[np.nanargmin(np.abs(fnr - fpr))]
    
    return eer_value, eer_threshold


def plot_roc_curve(y_true, y_scores, save_path=None):
    """ROC Curve 그리기"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    eer_value, eer_threshold = calculate_eer(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    
    # ROC Curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # EER Point
    eer_index = np.nanargmin(np.abs((1 - tpr) - fpr))
    plt.plot(fpr[eer_index], tpr[eer_index], 'ro', markersize=10, 
             label=f'EER = {eer_value:.4f} (threshold={eer_threshold:.4f})')
    
    # Random Classifier
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Anomaly Score Attention Model', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ROC Curve 저장: {save_path}")
    
    plt.close()
    
    return roc_auc, eer_value, eer_threshold


def plot_pr_curve(y_true, y_scores, save_path=None):
    """Precision-Recall Curve 그리기"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Baseline (클래스 비율)
    baseline = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], color='red', lw=2, linestyle='--', 
             label=f'Baseline ({baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Anomaly Score Attention Model', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  PR Curve 저장: {save_path}")
    
    plt.close()
    
    return pr_auc


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Confusion Matrix 그리기"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                cbar_kws={'label': 'Count'})
    
    # Normalized 값도 표시
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'({cm_normalized[i, j]:.2%})',
                    ha='center', va='center', color='red', fontsize=10)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix - Anomaly Score Attention Model', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Confusion Matrix 저장: {save_path}")
    
    plt.close()
    
    return cm


def plot_score_distribution(y_true, y_scores, save_path=None):
    """예측 점수 분포 그리기"""
    normal_scores = y_scores[y_true == 0]
    abnormal_scores = y_scores[y_true == 1]
    
    plt.figure(figsize=(12, 6))
    plt.hist(normal_scores, bins=50, alpha=0.7, color='blue', label='Normal')
    plt.hist(abnormal_scores, bins=50, alpha=0.7, color='red', label='Abnormal')
    
    plt.xlabel('Prediction Score (Abnormal Probability)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Score Distribution - Anomaly Score Attention Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Score Distribution 저장: {save_path}")
    
    plt.close()


def load_stgnf_models(device):
    """부위별 STG-NF 모델 로드"""
    from model_builder import create_stgnf_model
    
    print_section("부위별 STG-NF 모델 로드")
    
    stg_nf_models = {}
    
    pbar = tqdm(Config.Joint.BODY_PARTS, desc="Loading STG-NF models", unit="model")
    for part_name in pbar:
        pbar.set_postfix({'Current': part_name})
        
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        if joint_subset is None:
            num_joints = 18
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
        print(f"\n [{part_name}] 로드 완료")
    
    print(f"\n총 {len(stg_nf_models)}개 부위 모델 로드 완료\n")
    return stg_nf_models


def get_sample_data(device):
    """샘플 데이터 생성 (Feature 차원 계산용)"""
    print("[INFO] 샘플 데이터 생성 중...")
    
    sample_data_dict = {}
    for part_name in Config.Joint.BODY_PARTS:
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        if joint_subset is None:
            num_joints = 18
        else:
            num_joints = len(joint_subset)
        
        sample_data = torch.randn(
            1, 
            Config.STGNF.IN_CHANNELS, 
            Config.Data.SEG_LEN, 
            num_joints
        ).to(device)
        
        sample_data_dict[part_name] = sample_data
    
    return sample_data_dict


def evaluate_model(model, dataloader_dict, device, eval_threshold=0.7):
    """모델 평가"""
    model.eval()
    
    all_gt_labels = []
    all_pred_scores = []
    
    total_batches = len(dataloader_dict[Config.Joint.BODY_PARTS[0]])
    
    with torch.no_grad():
        batch_iter = tqdm(
            zip(*[dataloader_dict[part] for part in Config.Joint.BODY_PARTS]),
            total=total_batches,
            desc="Evaluating",
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
            
            # Forward
            logits = model(x_dict)
            probs = torch.softmax(logits, dim=1)
            abnormal_probs = probs[:, 1]  # Abnormal class 확률
            
            all_gt_labels.extend(labels.cpu().numpy())
            all_pred_scores.extend(abnormal_probs.cpu().numpy())
    
    all_gt_labels = np.array(all_gt_labels)
    all_pred_scores = np.array(all_pred_scores)
    
    return all_gt_labels, all_pred_scores


def main():
    print("\n" + "="*80)
    print("Anomaly Score Attention 분류기 평가")
    print("="*80 + "\n")
    
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='anomaly_attention_classifier_best.pth',
                       help='체크포인트 파일명')
    parser.add_argument('--results_dir', type=str, 
                       default=None,
                       help='결과 저장 디렉토리')
    args = vars(parser.parse_args())
    
    # 결과 저장 디렉토리
    if args['results_dir'] is None:
        args['results_dir'] = os.path.join(
            Config.Path.RESULTS_DIR, 
            'anomaly_attention_evaluation'
        )
    os.makedirs(args['results_dir'], exist_ok=True)
    
    print(f"[INFO] 결과 저장 디렉토리: {args['results_dir']}\n")
    
    # Seed 설정
    set_seed(Config.Train.SEED)
    
    # Device 설정
    device = torch.device(Config.Train.DEVICE)
    print(f"[INFO] Using device: {device}\n")
    
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
    
    # 4. 체크포인트 로드
    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, args['checkpoint'])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 없음: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] 체크포인트 로드: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}\n")
    
    model.eval()
    
    # 5. 테스트 데이터셋 로드
    print_section("테스트 데이터셋 로드")
    
    test_datasets_dict = {}
    for part_name in Config.Joint.BODY_PARTS:
        dataset = ShopliftingDataset(
            json_path=Config.Path.TEST_JSON,
            skeleton_base_path=Config.Path.TEST_DATA_DIR,
            joint_subset=Config.Joint.JOINT_SUBSET_MAP[part_name],
            seg_len=Config.Data.SEG_LEN,
            seg_stride=Config.Data.EVAL_STRIDE,  # 평가용 stride (1)
            normalize=Config.Data.NORMALIZE,
            apply_augmentation=False,
            vid_res=Config.Data.VID_RES,
            use_cache=False,  # 평가 시에는 캐싱 비활성화 (메모리 절약)
            load_per_batch=False,
            preprocess_cache=False  # 평가 시에는 전처리 캐싱도 비활성화
        )
        test_datasets_dict[part_name] = dataset
        print(f"  [{part_name}] samples: {len(dataset)}")
    
    # DataLoader 생성 (메모리 효율적 설정)
    test_dataloaders_dict = {}
    for part_name in Config.Joint.BODY_PARTS:
        dataloader = DataLoader(
            test_datasets_dict[part_name],
            batch_size=Config.Attention.BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # 평가 시에는 단일 프로세스 (Windows 멀티프로세싱 이슈 회피)
            pin_memory=False,  # 평가 시에는 pin_memory 비활성화 (메모리 절약)
            prefetch_factor=None,
            persistent_workers=False
        )
        test_dataloaders_dict[part_name] = dataloader
    
    # 6. 평가
    print_section("모델 평가")
    eval_threshold = Config.Train.EVAL_THRESHOLD
    
    all_gt_labels, all_pred_scores = evaluate_model(
        model, test_dataloaders_dict, device, eval_threshold
    )
    
    # 7. 결과 시각화
    print_section("결과 시각화")
    
    # 1. ROC Curve
    roc_auc, eer_value, eer_threshold = plot_roc_curve(
        all_gt_labels, all_pred_scores,
        save_path=os.path.join(args['results_dir'], 'roc_curve.png')
    )
    
    # 2. PR Curve
    pr_auc = plot_pr_curve(
        all_gt_labels, all_pred_scores,
        save_path=os.path.join(args['results_dir'], 'pr_curve.png')
    )
    
    # 3. Confusion Matrix (threshold=Config)
    all_preds = (all_pred_scores > eval_threshold).astype(int)
    cm = plot_confusion_matrix(
        all_gt_labels, all_preds,
        save_path=os.path.join(args['results_dir'], 'confusion_matrix.png')
    )
    
    # 4. Score Distribution
    plot_score_distribution(
        all_gt_labels, all_pred_scores,
        save_path=os.path.join(args['results_dir'], 'score_distribution.png')
    )
    
    # 5. Confusion Matrix (EER threshold)
    all_preds_eer = (all_pred_scores > eer_threshold).astype(int)
    cm_eer = plot_confusion_matrix(
        all_gt_labels, all_preds_eer,
        save_path=os.path.join(args['results_dir'], 'confusion_matrix_eer.png')
    )
    
    # 결과 출력
    print("\n" + "="*80)
    print("최종 평가 결과 - Anomaly Score Attention Model")
    print("="*80)
    
    accuracy = (all_preds == all_gt_labels).mean() * 100
    accuracy_eer = (all_preds_eer == all_gt_labels).mean() * 100
    
    print(f"\n[Threshold = {eval_threshold} (Config)]")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  AUC-ROC: {roc_auc:.4f}")
    print(f"  AUC-PR: {pr_auc:.4f}")
    
    print(f"\n[Threshold = {eer_threshold:.4f} (EER)]")
    print(f"  Accuracy: {accuracy_eer:.2f}%")
    print(f"  EER: {eer_value:.4f}")
    
    # Classification Report
    print(f"\n[Classification Report - Threshold={eval_threshold} (Config)]")
    print(classification_report(all_gt_labels, all_preds, 
                                target_names=['Normal', 'Abnormal'],
                                digits=4))
    
    print(f"\n[Classification Report - Threshold={eer_threshold:.4f} (EER)]")
    print(classification_report(all_gt_labels, all_preds_eer,
                                target_names=['Normal', 'Abnormal'],
                                digits=4))
    
    # 결과를 JSON으로 저장
    results = {
        f'threshold_{eval_threshold}': {
            'threshold': float(eval_threshold),
            'accuracy': float(accuracy),
            'auc_roc': float(roc_auc),
            'auc_pr': float(pr_auc),
            'confusion_matrix': cm.tolist()
        },
        'threshold_eer': {
            'threshold': float(eer_threshold),
            'eer': float(eer_value),
            'accuracy': float(accuracy_eer),
            'confusion_matrix': cm_eer.tolist()
        }
    }
    
    results_json_path = os.path.join(args['results_dir'], 'evaluation_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n[INFO] 평가 결과 JSON 저장: {results_json_path}")
    print(f"[INFO] 모든 결과가 {args['results_dir']} 에 저장되었습니다.\n")


if __name__ == "__main__":
    main()
