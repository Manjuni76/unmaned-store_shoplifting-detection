"""
통합 학습 및 평가 파이프라인
1. STG-NF 부위별 학습 (체크포인트 없으면 실행)
2. Attention 분류기 학습 (체크포인트 없으면 실행)
3. 테스트 데이터 평가
"""

import os
import sys
import time
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc as compute_auc
from tqdm import tqdm

from args import Config
from dataset import ShopliftingDataset
from model_builder import create_stgnf_model
from attention_classifier import create_attention_classifier
from utils_train import set_seed, print_section, load_checkpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def check_stgnf_checkpoints():
    """STG-NF 체크포인트 존재 여부 확인"""
    missing_parts = []
    existing_parts = []
    
    for part_name in Config.Joint.BODY_PARTS:
        checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, f'stgnf_{part_name}.pth')
        if os.path.exists(checkpoint_path):
            existing_parts.append(part_name)
        else:
            missing_parts.append(part_name)
    
    return existing_parts, missing_parts


def check_attention_checkpoint():
    """Attention 분류기 체크포인트 존재 여부 확인"""
    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, 'attention_classifier_final.pth')
    return os.path.exists(checkpoint_path), checkpoint_path


def run_stgnf_training():
    """STG-NF 학습 스크립트 실행"""
    print_section("STG-NF 부위별 학습 실행")
    print("실행 명령: python train_stgnf_parts.py")
    
    result = subprocess.run([sys.executable, 'train_stgnf_parts.py'], 
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        raise RuntimeError("STG-NF 학습 실패")
    
    print("\nSTG-NF 학습 완료!")


def run_attention_training():
    """Attention 분류기 학습 스크립트 실행"""
    print_section("Attention 분류기 학습 실행")
    print("실행 명령: python train_attention_classifier.py")
    
    result = subprocess.run([sys.executable, 'train_attention_classifier.py'], 
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        raise RuntimeError("Attention 분류기 학습 실패")
    
    print("\nAttention 분류기 학습 완료!")


def load_trained_models(device):
    """학습된 모델 로드"""
    print_section("학습된 모델 로드")
    
    stg_nf_models = {}
    
    for part_name in Config.Joint.BODY_PARTS:
        checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, f'stgnf_{part_name}.pth')
        
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
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
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        stg_nf_models[part_name] = model
        print(f"  [{part_name}] STG-NF 로드 완료")
    
    print("\n샘플 데이터 생성 중...")
    sample_data_dict = {}
    
    for part_name in Config.Joint.BODY_PARTS:
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        temp_dataset = ShopliftingDataset(
            json_path=Config.Path.TEST_JSON,
            skeleton_base_path=Config.Path.TEST_DATA_DIR,
            seg_len=Config.Data.SEG_LEN,
            seg_stride=Config.Data.EVAL_STRIDE,
            joint_subset=joint_subset,
            normalize=Config.Data.NORMALIZE,
            apply_augmentation=False,
            vid_res=Config.Data.VID_RES
        )
        
        if len(temp_dataset) > 0:
            sample, _ = temp_dataset[0]
            sample_data_dict[part_name] = sample.unsqueeze(0).to(device)
    
    # Attention 분류기 로드
    attention_checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, 'attention_classifier_final.pth')
    
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
    
    checkpoint = torch.load(attention_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nAttention 분류기 로드 완료")
    
    return model


def evaluate_on_test(model, device):
    """테스트 데이터로 평가"""
    print_section("테스트 데이터 평가")
    
    test_datasets = {}
    test_loaders = {}
    
    for part_name in Config.Joint.BODY_PARTS:
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        dataset = ShopliftingDataset(
            json_path=Config.Path.TEST_JSON,
            skeleton_base_path=Config.Path.TEST_DATA_DIR,
            seg_len=Config.Data.SEG_LEN,
            seg_stride=Config.Data.EVAL_STRIDE,
            joint_subset=joint_subset,
            normalize=Config.Data.NORMALIZE,
            apply_augmentation=False,
            vid_res=Config.Data.VID_RES
        )
        
        loader = DataLoader(
            dataset,
            batch_size=Config.Attention.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.Train.NUM_WORKERS,
            pin_memory=True
        )
        
        test_datasets[part_name] = dataset
        test_loaders[part_name] = loader
    
    print(f"테스트 데이터 로드 완료")
    print(f"총 프레임 수: {len(test_datasets[Config.Joint.BODY_PARTS[0]])}")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n평가 중...")
    with torch.no_grad():
        for batch_data in tqdm(zip(*[test_loaders[part] for part in Config.Joint.BODY_PARTS]), 
                               total=len(test_loaders[Config.Joint.BODY_PARTS[0]])):
            x_dict = {}
            labels = None
            
            for i, part in enumerate(Config.Joint.BODY_PARTS):
                data, label = batch_data[i]
                x_dict[part] = data.to(device)
                if labels is None:
                    labels = label.to(device)
            
            logits = model(x_dict)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    auc_roc = roc_auc_score(all_labels, all_probs)
    
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auc_pr = compute_auc(recall, precision)
    
    print_section("평가 결과")
    print(f"총 프레임 수: {len(all_labels):,}")
    print(f"  - 정상: {sum([1 for l in all_labels if l == 0]):,}")
    print(f"  - 이상: {sum([1 for l in all_labels if l == 1]):,}")
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    
    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def main():
    """메인 함수"""
    print_section("통합 학습 및 평가 파이프라인")
    
    set_seed(Config.Train.SEED)
    
    device = torch.device(Config.Train.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    Config.Path.create_dirs()
    
    print_section("Step 1: STG-NF 체크포인트 확인")
    
    existing_parts, missing_parts = check_stgnf_checkpoints()
    
    print(f"존재하는 STG-NF 모델: {existing_parts}")
    print(f"없는 STG-NF 모델: {missing_parts}")
    
    if missing_parts:
        print(f"\n{len(missing_parts)}개 부위의 STG-NF 모델이 없습니다.")
        print("STG-NF 학습을 시작합니다...\n")
        run_stgnf_training()
    else:
        print("\n모든 STG-NF 모델이 존재합니다. 학습 스킵.")
    
    print_section("Step 2: Attention 분류기 체크포인트 확인")
    
    attention_exists, attention_path = check_attention_checkpoint()
    
    if attention_exists:
        print(f"Attention 분류기 존재: {attention_path}")
        print("Attention 학습 스킵.")
    else:
        print("Attention 분류기가 없습니다.")
        print("Attention 학습을 시작합니다...\n")
        run_attention_training()
    
    print_section("Step 3: 모델 로드 및 테스트 평가")
    
    model = load_trained_models(device)
    
    results = evaluate_on_test(model, device)
    
    print_section("파이프라인 완료!")
    print("모든 단계가 성공적으로 완료되었습니다.")
    print(f"\n최종 성능:")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  AUC-ROC: {results['auc_roc']:.4f}")
    print(f"  AUC-PR: {results['auc_pr']:.4f}")
    
    print(f"\n저장된 모델:")
    print(f"  - STG-NF 모델: {Config.Path.CHECKPOINT_DIR}/stgnf_*.pth")
    print(f"  - Attention 모델: {Config.Path.CHECKPOINT_DIR}/attention_classifier_final.pth")
    
    print(f"\n다음 단계:")
    print("  - 자세한 평가: python evaluate_model.py")
    print("  - 재학습: 체크포인트 파일을 삭제 후 다시 실행")


if __name__ == "__main__":
    main()
