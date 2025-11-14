"""
학습된 모델 평가 스크립트
- AUC-ROC 곡선
- AUC-PR 곡선
- EER (Equal Error Rate)
- 혼동 행렬 (Confusion Matrix)
- 프레임별 평가
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report,
    roc_auc_score
)
from tqdm import tqdm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 필요한 클래스/함수 import
from dataset import ShopliftingDataset
from model_builder import Multi_STG_NF_with_Attention, create_stgnf_model
from attention_classifier import create_attention_classifier
from train_pipeline import set_seed
from args import Config

# JOINT_SUBSET_MAP 가져오기
JOINT_SUBSET_MAP = Config.Joint.JOINT_SUBSET_MAP

try:
    from models.STG_NF.model_pose import STG_NF
    print("[SUCCESS] STG-NF 모델 import 성공")
except ImportError as e:
    print(f"[ERROR] STG-NF 모델 import 실패: {e}")
    STG_NF = None


def calculate_eer(y_true, y_scores):
    """
    EER (Equal Error Rate) 계산
    FAR(False Accept Rate) = FPR(False Positive Rate)과 
    FRR(False Reject Rate) = FNR(False Negative Rate)이 같아지는 지점
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # FAR과 FRR이 가장 가까운 지점 찾기
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer_value = fpr[np.nanargmin(np.abs(fnr - fpr))]
    
    return eer_value, eer_threshold


def plot_roc_curve(y_true, y_scores, save_path='results/roc_curve.png'):
    """ROC 곡선 그리기"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer_value, eer_threshold = calculate_eer(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    # EER 지점 표시
    eer_idx = np.nanargmin(np.abs((1-tpr) - fpr))
    plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=10, 
             label=f'EER = {eer_value:.4f} (threshold={eer_threshold:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('ROC Curve - Shoplifting Detection', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] ROC curve 저장: {save_path}")
    plt.close()
    
    return roc_auc, eer_value, eer_threshold


def plot_pr_curve(y_true, y_scores, save_path='results/pr_curve.png'):
    """Precision-Recall 곡선 그리기"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Baseline (클래스 비율)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='red', linestyle='--', lw=2,
                label=f'Baseline (Positive ratio = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve - Shoplifting Detection', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] PR curve 저장: {save_path}")
    plt.close()
    
    return pr_auc


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix - Shoplifting Detection', fontsize=16)
    
    # 각 셀에 비율도 표시
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f'({cm_normalized[i, j]:.2%})',
                    ha='center', va='center', fontsize=10, color='gray')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] Confusion matrix 저장: {save_path}")
    plt.close()
    
    return cm


def plot_score_distribution(y_true, y_scores, save_path='results/score_distribution.png'):
    """정상/이상 클래스별 예측 점수 분포"""
    normal_scores = y_scores[y_true == 0]
    abnormal_scores = y_scores[y_true == 1]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(normal_scores, bins=50, alpha=0.7, color='blue', label='Normal')
    plt.hist(abnormal_scores, bins=50, alpha=0.7, color='red', label='Abnormal')
    plt.xlabel('Prediction Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Score Distribution by Class', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([normal_scores, abnormal_scores], 
                labels=['Normal', 'Abnormal'],
                showfliers=True)
    plt.ylabel('Prediction Score', fontsize=12)
    plt.title('Score Distribution (Box Plot)', fontsize=14)
    plt.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] Score distribution 저장: {save_path}")
    plt.close()


def load_models(checkpoint_dir, device):
    """부위별 STG-NF 모델 로드"""
    body_parts = ['head', 'arms', 'body', 'legs', 'all']
    stg_nf_models = {}
    
    print("\n" + "="*80)
    print("부위별 STG-NF 모델 로드 중...")
    print("="*80)
    
    for part in body_parts:
        checkpoint_path = os.path.join(checkpoint_dir, f'stgnf_{part}.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"체크포인트 파일 없음: {checkpoint_path}")
        
        # 더미 데이터로 모델 구조 파악
        if part == 'head':
            V = 5
        elif part == 'arms':
            V = 6
        elif part == 'body':
            V = 5
        elif part == 'legs':
            V = 6
        else:  # all
            V = 18
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 체크포인트 형식 확인
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # create_stgnf_model을 사용하여 모델 생성 (Config 값 사용)
        model = create_stgnf_model(
            in_channels=Config.STGNF.IN_CHANNELS,
            hidden_dim=Config.STGNF.HIDDEN_CHANNELS,
            num_layers=Config.STGNF.K,
            num_frames=Config.Data.SEG_LEN,  # Config에서 가져옴 (24)
            num_joints=V,
            device=device,
            subset_idx=JOINT_SUBSET_MAP[part]
        )
        
        # 상태 로드
        model.load_state_dict(state_dict)
        model.eval()
        
        stg_nf_models[part] = model
        print(f"  [{part}] 모델 로드 완료")
    
    print("\n부위별 STG-NF 모델 로드 완료 ✓\n")
    return stg_nf_models


def evaluate_model(args):
    """모델 평가 및 결과 저장"""
    print("\n" + "="*80)
    print("모델 평가 시작")
    print("="*80)
    
    # 시드 설정
    set_seed(args['seed'])
    
    # 부위별 STG-NF 모델 로드
    stg_nf_models = load_models(args['checkpoint_dir'], args['device'])
    
    # 부위별 테스트 데이터셋 로드 (stride=1로 모든 프레임 평가)
    test_datasets = {}
    for part in stg_nf_models.keys():
        test_datasets[part] = ShopliftingDataset(
            json_path=args['test_json'],
            skeleton_base_path=args['test_skeleton_path'],
            seg_len=args['seg_len'],
            seg_stride=1,  # stride=1로 모든 프레임 평가
            joint_subset=JOINT_SUBSET_MAP[part],
            normalize=True,
            apply_augmentation=False,
            vid_res=args['vid_res']
        )
    
    print(f"[DATASET] 부위별 테스트 데이터셋 로드 완료 (stride=1)")
    
    # 샘플 데이터로 통합 모델 생성
    sample_data_dict = {}
    for part, dataset in test_datasets.items():
        sample_data, _ = dataset[0]
        sample_data_dict[part] = sample_data.unsqueeze(0).to(args['device'])
    
    # Multi STG-NF + Attention 모델 생성 및 로드
    print("[INFO] Attention 분류기 생성 중...")
    full_model = create_attention_classifier(
        stg_nf_models_dict=stg_nf_models,
        sample_data_dict=sample_data_dict,
        num_classes=Config.Attention.NUM_CLASSES,
        embed_dim=Config.Attention.EMBED_DIM,
        num_heads=Config.Attention.NUM_HEADS,
        num_encoder_layers=Config.Attention.NUM_ENCODER_LAYERS,
        dropout=Config.Attention.DROPOUT,
        device=args['device']
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(args['model_checkpoint'], map_location=args['device'])
    if 'model_state_dict' in checkpoint:
        full_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        full_model.load_state_dict(checkpoint)
    full_model.eval()
    print(f"[LOAD] 통합 모델 로드 완료: {args['model_checkpoint']}\n")
    
    # GT 파일 경로
    gt_dir = args['gt_dir']
    
    # 프레임별 평가
    print("프레임별 예측 수행 중...")
    video_predictions = {}
    
    num_samples = len(test_datasets[list(test_datasets.keys())[0]])
    batch_size = args['batch_size']
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            # 부위별 배치 데이터 수집
            x_dict = {}
            metadata_list = []
            
            for part, dataset in test_datasets.items():
                batch_data = []
                for idx in range(start_idx, end_idx):
                    data, label = dataset[idx]
                    batch_data.append(data)
                    if part == list(test_datasets.keys())[0]:
                        metadata_list.append(dataset.segment_metadata[idx])
                
                x_dict[part] = torch.stack(batch_data).to(args['device'])
            
            logits = full_model(x_dict)
            scores = torch.softmax(logits, dim=1)[:, 1]  # 이상 클래스 확률
            
            # 프레임별로 예측 저장
            for i, metadata in enumerate(metadata_list):
                filename = metadata['filename']
                center_frame = metadata['center_frame']
                pred_score = scores[i].item()
                
                if filename not in video_predictions:
                    video_predictions[filename] = {}
                
                video_predictions[filename][center_frame] = pred_score
    
    # GT 파일 로드하고 프레임별 비교
    print("\nGT와 비교 중...")
    all_gt_labels = []
    all_pred_scores = []
    
    for filename in tqdm(video_predictions.keys(), desc="Loading GT"):
        gt_filename = filename.replace('.mp4', '.npy')
        gt_path = os.path.join(gt_dir, gt_filename)
        
        if not os.path.exists(gt_path):
            print(f"[WARNING] GT 파일 없음: {gt_path}")
            continue
        
        gt_labels = np.load(gt_path)
        
        frame_indices = sorted(video_predictions[filename].keys())
        for frame_idx in frame_indices:
            if frame_idx < len(gt_labels):
                all_gt_labels.append(gt_labels[frame_idx])
                all_pred_scores.append(video_predictions[filename][frame_idx])
    
    all_gt_labels = np.array(all_gt_labels)
    all_pred_scores = np.array(all_pred_scores)
    
    print(f"\n총 평가 프레임 수: {len(all_gt_labels)}")
    print(f"정상 프레임: {np.sum(all_gt_labels == 0)} ({np.sum(all_gt_labels == 0)/len(all_gt_labels)*100:.2f}%)")
    print(f"이상 프레임: {np.sum(all_gt_labels == 1)} ({np.sum(all_gt_labels == 1)/len(all_gt_labels)*100:.2f}%)")
    
    # 메트릭 계산
    print("\n" + "="*80)
    print("메트릭 계산 및 시각화")
    print("="*80)
    
    # 1. ROC Curve & AUC-ROC & EER
    roc_auc, eer_value, eer_threshold = plot_roc_curve(
        all_gt_labels, all_pred_scores, 
        save_path=os.path.join(args['results_dir'], 'roc_curve.png')
    )
    
    # 2. PR Curve & AUC-PR
    pr_auc = plot_pr_curve(
        all_gt_labels, all_pred_scores,
        save_path=os.path.join(args['results_dir'], 'pr_curve.png')
    )
    
    # 3. Confusion Matrix (threshold from Config.Train.EVAL_THRESHOLD)
    from args import Config
    eval_threshold = Config.Train.EVAL_THRESHOLD
    all_preds = (all_pred_scores > eval_threshold).astype(int)
    cm = plot_confusion_matrix(
        all_gt_labels, all_preds,
        save_path=os.path.join(args['results_dir'], f'confusion_matrix_th{eval_threshold}.png')
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
    print("최종 평가 결과")
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
        },
        'total_frames': int(len(all_gt_labels)),
        'normal_frames': int(np.sum(all_gt_labels == 0)),
        'abnormal_frames': int(np.sum(all_gt_labels == 1))
    }
    
    results_json_path = os.path.join(args['results_dir'], 'evaluation_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] 평가 결과 저장: {results_json_path}")
    
    print("\n" + "="*80)
    print("평가 완료!")
    print("="*80)


def main():
    # 베이스 경로
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    args = {
        'test_json': os.path.join(base_dir, 'data_split', 'output', 'test_data.json'),
        'test_skeleton_path': os.path.join(base_dir, 'data', 'test_data_skeleton_data'),
        'gt_dir': os.path.join(base_dir, 'data', 'gt', 'test_gt'),
        'checkpoint_dir': os.path.join(base_dir, 'ai_model', 'checkpoints'),
        'model_checkpoint': os.path.join(base_dir, 'ai_model', 'checkpoints', 'attention_classifier_final.pth'),
        'results_dir': os.path.join(base_dir, 'ai_model', 'results'),
        'seg_len': Config.Data.SEG_LEN,  # Config에서 가져옴 (24)
        'batch_size': 32,
        'vid_res': [1920, 1080],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }
    
    # 결과 디렉토리 생성
    os.makedirs(args['results_dir'], exist_ok=True)
    
    print(f"\n{'='*80}")
    print("절도 행위 탐지 모델 평가")
    print(f"{'='*80}")
    print(f"Device: {args['device']}")
    print(f"Test JSON: {args['test_json']}")
    print(f"Model Checkpoint: {args['model_checkpoint']}")
    print(f"Results Directory: {args['results_dir']}")
    
    evaluate_model(args)


if __name__ == '__main__':
    main()
