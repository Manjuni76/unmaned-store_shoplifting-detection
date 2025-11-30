"""
전체 파이프라인 평가 스크립트 V2
STG-NF (부위별, 잠재벡터 z) → Attention Classifier → 최종 예측
혼동행렬, AUC-ROC, AUC-PR, EER 계산
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, 
    roc_auc_score, 
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# 설정 import
from args import Config
from datasets.dataset_folder_scan import FolderScanDataset
from models.stgnf_loader import load_all_stgnf_models
from models.model_builder import create_attention_classifier
from datasets.train_utils import set_seed, print_section

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_attention_model(stgnf_models, sample_data_dict, device):
    """Attention Classifier 로드"""
    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, 'attention_fin.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Attention Classifier 없음: {checkpoint_path}")
        return None
    
    # Attention 모델 생성
    print("  Attention Classifier 생성 중...")
    model = create_attention_classifier(
        stg_nf_models_dict=stgnf_models,
        sample_data_dict=sample_data_dict,
        num_classes=Config.Attention.NUM_CLASSES,
        embed_dim=Config.Attention.EMBED_DIM,
        num_heads=Config.Attention.NUM_HEADS,
        num_encoder_layers=Config.Attention.NUM_ENCODER_LAYERS,
        dropout=Config.Attention.DROPOUT,
        device=str(device)
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Attention Classifier 로드 완료")
    print(f"    Epoch: {checkpoint['epoch']}")
    print(f"    Val F1: {checkpoint.get('val_f1', 0):.2f}%")
    print(f"    Val Acc: {checkpoint.get('val_acc', 0):.2f}%")
    
    return model


@torch.no_grad()
def predict_frame_level(model, dataset, device):
    from collections import defaultdict
    
    model.eval()
    
    # 비디오별 프레임별 점수 저장 (겹치는 부분 max 처리용)
    video_frame_scores = defaultdict(lambda: defaultdict(list))
    
    seg_len = Config.Data.SEG_LEN
    segment_metadata = dataset.segment_metadata
    
    # DataLoader로 세그먼트 예측
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)
    
    segment_idx = 0
    pbar = tqdm(dataloader, desc="  Predicting segments", leave=False)
    for data, labels in pbar:
        data = data.to(device)  # (B, C, T, V)
        batch_size = data.shape[0]
        
        # x_dict: 부위별 subset 슬라이싱
        x_dict = {}
        for part in model.part_names:
            subset = Config.Joint.JOINT_SUBSET_MAP[part]
            if subset is not None:
                x_dict[part] = data[:, :, :, subset]
            else:
                x_dict[part] = data
        
        # Forward pass
        logits = model(x_dict)  # (B, num_classes)
        probs = torch.softmax(logits, dim=1)  # (B, num_classes)
        abnormal_scores = probs[:, 1].cpu().numpy()  # positive class (abnormal) 확률
        
        # 각 세그먼트의 점수를 관련 프레임들에 할당
        for i in range(batch_size):
            if segment_idx >= len(segment_metadata):
                break
            
            meta = segment_metadata[segment_idx]
            filename = meta['filename']
            # filename에서 basename만 추출 (경로 제거)
            basename = os.path.basename(filename)
            video_id = basename.replace('.json', '')
            start_frame = meta['start_frame']
            
            # 이 세그먼트의 abnormal 점수를 해당 프레임들(24개)에 할당
            score = float(abnormal_scores[i])
            for frame_idx in range(start_frame, start_frame + seg_len):
                video_frame_scores[video_id][frame_idx].append(score)
            
            segment_idx += 1
    
    # 프레임별 점수 집계 (max 사용)
    print(f"\n  프레임별 점수 집계 중 (max)... 비디오 수: {len(video_frame_scores)}")
    video_frame_final = {}
    total_frames = 0
    for video_id, frame_dict in video_frame_scores.items():
        video_frame_final[video_id] = {}
        for frame_idx, scores in frame_dict.items():
            video_frame_final[video_id][frame_idx] = np.max(scores)  # 겹치는 부분 max
            total_frames += 1
    print(f"  총 프레임 수: {total_frames}")
    
    # GT 로드
    print("  GT 로드 중...")
    video_frame_gt = {}
    gt_not_found_count = 0
    for meta in segment_metadata:
        filename = meta['filename']
        # filename은 이미 확장자 없는 basename (dataset에서 .npy 제거한 상태)
        video_id = filename
        
        if video_id in video_frame_gt:
            continue
        
        # GT 파일명: filename + .npy
        gt_filename = filename + '.npy'
        gt_path = os.path.join(Config.Path.TEST_GT_DIR, gt_filename)
        
        if not os.path.exists(gt_path):
            if gt_not_found_count < 3:  # 처음 3개만 출력
                print(f"    GT 찾을 수 없음: {gt_path}")
            gt_not_found_count += 1
            continue
        
        gt_array = np.load(gt_path)
        frame_labels = {i: int(gt_array[i]) for i in range(len(gt_array))}
        video_frame_gt[video_id] = frame_labels
    
    if gt_not_found_count > 0:
        print(f"  GT 찾을 수 없음 총 {gt_not_found_count}개")
    print(f"  로드된 GT 비디오 수: {len(video_frame_gt)}")
    
    # 프레임별 GT와 예측 점수 매칭
    print(f"  프레임별 GT 매칭 중... (예측 비디오: {len(video_frame_final)}, GT 비디오: {len(video_frame_gt)})")
    all_frame_gts = []
    all_frame_scores = []
    
    for video_id, frame_dict in video_frame_final.items():
        if video_id not in video_frame_gt:
            print(f"    경고: {video_id}의 GT를 찾을 수 없음")
            continue
        
        gt_frames = video_frame_gt[video_id]
        
        for frame_idx, score in frame_dict.items():
            if frame_idx not in gt_frames:
                continue
            
            all_frame_gts.append(gt_frames[frame_idx])
            all_frame_scores.append(score)
    
    print(f"  매칭된 프레임 수: {len(all_frame_gts)}")
    
    if len(all_frame_gts) == 0:
        print("[ERROR] 매칭된 프레임이 없습니다!")
        print(f"  예측 비디오 IDs: {list(video_frame_final.keys())[:5]}")
        print(f"  GT 비디오 IDs: {list(video_frame_gt.keys())[:5]}")
        return None, None, None
    
    y_true = np.array(all_frame_gts)
    y_scores = np.array(all_frame_scores)
    
    # 0.5 threshold로 예측 라벨 생성
    y_pred = (y_scores > 0.5).astype(int)
    
    return y_true, y_pred, y_scores


def calculate_eer(y_true, y_scores):
    """EER (Equal Error Rate) 계산"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # FPR과 FNR의 차이가 최소인 지점
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold


def plot_confusion_matrix(cm, save_path):
    """혼동행렬 시각화"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  혼동행렬 저장: {save_path}")


def plot_roc_curve(y_true, y_scores, auc_score, save_path):
    """ROC 커브 시각화"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC-ROC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ROC 커브 저장: {save_path}")


def plot_pr_curve(y_true, y_scores, auc_score, save_path):
    """Precision-Recall 커브 시각화"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'AUC-PR = {auc_score:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  PR 커브 저장: {save_path}")


def evaluate_pipeline():
    """전체 파이프라인 평가"""
    print_section("전체 파이프라인 평가")
    
    # 시드 설정
    set_seed(Config.Train.SEED)
    
    # 디바이스 설정
    device = torch.device(Config.Train.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}\n")
    
    # 1. STG-NF 모델들 로드 (ActNorm 유지)
    print_section("1. STG-NF 모델 로드")
    stgnf_models = load_all_stgnf_models(device)
    
    if len(stgnf_models) == 0:
        print("[ERROR] STG-NF 모델이 없습니다!")
        return
    
    # 2. 테스트 데이터셋 (stride=1로 모든 프레임 커버)
    print_section("2. 테스트 데이터셋 로드")
    test_dataset = FolderScanDataset(
        skeleton_dir=Config.Path.TEST_DATA_DIR,
        gt_dir=Config.Path.TEST_GT_DIR,
        seg_len=Config.Data.SEG_LEN,
        seg_stride=1,  # stride=1로 모든 프레임 커버
        joint_subset=None,  # 전체 관절
        normalize=Config.Data.NORMALIZE,
        apply_augmentation=False,
        vid_res=Config.Data.VID_RES,
        use_cache=False,
        preprocess_cache=False
    )
    
    print(f"  총 세그먼트: {len(test_dataset)} (stride=1, 프레임별 평가)")
    
    # 3. 샘플 데이터 준비 (Feature 차원 계산용)
    print_section("3. Feature 차원 계산")
    temp_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    sample_batch = next(iter(temp_loader))[0][:1]  # (1, C, T, V)
    sample_data_dict = {}
    for part in stgnf_models.keys():
        subset = Config.Joint.JOINT_SUBSET_MAP[part]
        if subset is not None:
            sample_data_dict[part] = sample_batch[:, :, :, subset].to(device)
        else:
            sample_data_dict[part] = sample_batch.to(device)
    
    # 4. Attention Classifier 로드
    print_section("4. Attention Classifier 로드")
    attention_model = load_attention_model(stgnf_models, sample_data_dict, device)
    
    if attention_model is None:
        print("[ERROR] Attention Classifier 로드 실패!")
        return
    
    # 5. 프레임별 예측 (겹치는 부분 max 처리)
    print_section("5. 프레임별 예측 수행")
    y_true, y_pred, y_scores = predict_frame_level(attention_model, test_dataset, device)
    
    # 예측 실패 시 종료
    if y_true is None or len(y_true) == 0:
        print("[ERROR] 프레임별 예측 실패!")
        return
    
    # 6. 평가 지표 계산
    print_section("6. 평가 지표 계산")
    
    # 결과 확인
    print(f"  총 프레임 수: {len(y_true)}")
    print(f"  GT 분포: Normal={np.sum(y_true==0)}, Abnormal={np.sum(y_true==1)}")
    print(f"  예측 분포: Normal={np.sum(y_pred==0)}, Abnormal={np.sum(y_pred==1)}")
    print(f"  점수 범위: [{y_scores.min():.4f}, {y_scores.max():.4f}]")
    
    # 혼동행렬
    cm = confusion_matrix(y_true, y_pred)
    print(f"  혼동행렬 shape: {cm.shape}")
    
    if cm.size == 0:
        print("ERROR: 혼동행렬이 비어있습니다!")
        return
    
    if cm.shape[0] < 2 or cm.shape[1] < 2:
        print(f"ERROR: 혼동행렬 크기 부족: {cm.shape}")
        return
    
    tn, fp, fn, tp = cm.ravel()
    
    # 정확도, 정밀도, 재현율, F1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUC-ROC, AUC-PR
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    
    # EER
    eer, eer_threshold = calculate_eer(y_true, y_scores)
    
    # 결과 출력
    print("\n" + "="*70)
    print("혼동행렬:")
    print(f"              Predicted")
    print(f"              Normal  Abnormal")
    print(f"Actual Normal   {tn:6d}  {fp:6d}")
    print(f"       Abnormal {fn:6d}  {tp:6d}")
    print("="*70)
    
    print(f"\n분류 성능:")
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    
    print(f"\nAUC 점수:")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"  AUC-PR    : {auc_pr:.4f}")
    
    print(f"\nEER (Equal Error Rate):")
    print(f"  EER       : {eer*100:.2f}%")
    print(f"  Threshold : {eer_threshold:.4f}")
    
    print("="*70 + "\n")
    
    # 7. 시각화
    print_section("7. 결과 시각화")
    results_dir = os.path.join(Config.Path.CHECKPOINT_DIR, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 혼동행렬
    plot_confusion_matrix(cm, os.path.join(results_dir, 'confusion_matrix.png'))
    
    # ROC 커브
    plot_roc_curve(y_true, y_scores, auc_roc, os.path.join(results_dir, 'roc_curve.png'))
    
    # PR 커브
    plot_pr_curve(y_true, y_scores, auc_pr, os.path.join(results_dir, 'pr_curve.png'))
    
    # 8. 결과 저장
    print_section("8. 결과 저장")
    results = {
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'eer': float(eer),
            'eer_threshold': float(eer_threshold)
        },
        'total_segments': int(len(y_true)),
        'normal_segments': int(np.sum(y_true == 0)),
        'abnormal_segments': int(np.sum(y_true == 1))
    }
    
    results_path = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  평가 결과 저장: {results_path}")
    print(f"  결과 디렉토리: {results_dir}\n")


if __name__ == "__main__":
    evaluate_pipeline()
