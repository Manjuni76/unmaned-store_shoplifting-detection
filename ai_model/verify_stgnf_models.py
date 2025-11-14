"""
STG-NF 모델 검증 스크립트
seg_len=48로 학습된 STG-NF 모델들이 정상/이상 데이터를 제대로 구분하는지 검증
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 설정 import
from args import Config
from dataset import ShopliftingDataset
from model_builder import create_stgnf_model
from utils_train import set_seed, print_section

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_stgnf_model(part_name, joint_subset, device):
    """부위별 STG-NF 모델 로드"""
    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, f'stgnf_{part_name}.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] 모델 파일 없음: {checkpoint_path}")
        return None
    
    # 관절 수 계산
    num_joints = len(joint_subset) if joint_subset is not None else 18
    
    # 모델 생성
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
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # ActNorm 초기화를 위한 dummy forward pass
    model.train()
    with torch.no_grad():
        dummy_input = torch.randn(1, Config.STGNF.IN_CHANNELS, Config.Data.SEG_LEN, num_joints).to(device)
        _ = model(dummy_input)
    
    model.eval()
    
    print(f"  [{part_name}] 모델 로드 완료 (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    
    return model


def calculate_nll_scores(model, dataloader, device, max_samples=None):
    """데이터셋의 NLL 점수 계산 (프레임별 정상/이상 구분)"""
    nll_scores = []
    sample_count = 0
    
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="  Calculating NLL", leave=False):
            if max_samples and sample_count >= max_samples:
                break
            
            data = data.to(device)  # (B, C=3, T, V)
            
            # Forward pass
            z, nll = model(data)
            
            nll_scores.extend(nll.cpu().numpy().tolist())
            sample_count += len(nll)
    
    return np.array(nll_scores if not max_samples else nll_scores[:max_samples])


def verify_stgnf_models():
    """STG-NF 모델들 검증"""
    print_section("STG-NF 모델 검증")
    
    # 모델 학습 시 사용한 SEG_LEN 확인
    # 현재 설정값 사용 (args.py에서 SEG_LEN=24)
    seg_len_for_stgnf = Config.Data.SEG_LEN
    eval_stride = Config.Data.EVAL_STRIDE
    
    print(f"SEG_LEN: {seg_len_for_stgnf}")
    print(f"EVAL_STRIDE: {eval_stride}")
    print(f"검증할 부위: {Config.Joint.BODY_PARTS}")
    print(f"샘플 수: Normal 1000개, Abnormal 1000개\n")
    
    # 시드 설정
    set_seed(Config.Train.SEED)
    
    # 디바이스 설정
    device = torch.device(Config.Train.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}\n")
    
    # 결과 저장용
    results = {}
    
    # 부위별 검증
    for part_name in Config.Joint.BODY_PARTS:
        print_section(f"[{part_name.upper()}] 부위 검증")
        
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        # 1. STG-NF 모델 로드
        print(f"\n[{part_name}] STG-NF 모델 로드 중...")
        model = load_stgnf_model(part_name, joint_subset, device)
        
        if model is None:
            print(f"[{part_name}] 모델 로드 실패. 스킵.\n")
            continue
        
        # 2. 정상 프레임 데이터 로드 (Test Set, filter_label='normal')
        print(f"\n[{part_name}] 정상 프레임 데이터 로드 중...")
        normal_dataset = ShopliftingDataset(
            json_path=Config.Path.TEST_JSON,
            skeleton_base_path=Config.Path.TEST_DATA_DIR,
            seg_len=seg_len_for_stgnf,
            seg_stride=eval_stride,
            joint_subset=joint_subset,
            normalize=Config.Data.NORMALIZE,
            apply_augmentation=False,
            vid_res=Config.Data.VID_RES,
            use_cache=False,
            load_per_batch=False,
            preprocess_cache=False,
            filter_label='normal'  # 중앙 프레임이 정상인 세그먼트만
        )
        
        normal_dataloader = DataLoader(
            normal_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"  정상 프레임 세그먼트 수: {len(normal_dataset)}")
        
        # 3. 이상 프레임 데이터 로드 (Test Set, filter_label='abnormal')
        print(f"\n[{part_name}] 이상 프레임 데이터 로드 중...")
        abnormal_dataset = ShopliftingDataset(
            json_path=Config.Path.TEST_JSON,
            skeleton_base_path=Config.Path.TEST_DATA_DIR,
            seg_len=seg_len_for_stgnf,
            seg_stride=eval_stride,
            joint_subset=joint_subset,
            normalize=Config.Data.NORMALIZE,
            apply_augmentation=False,
            vid_res=Config.Data.VID_RES,
            use_cache=False,
            load_per_batch=False,
            preprocess_cache=False,
            filter_label='abnormal'  # 중앙 프레임이 이상인 세그먼트만
        )
        
        abnormal_dataloader = DataLoader(
            abnormal_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"  이상 프레임 세그먼트 수: {len(abnormal_dataset)}")
        
        # 4. NLL 점수 계산 (프레임별)
        print(f"\n[{part_name}] NLL 점수 계산 중...")
        
        print(f"  정상 프레임 NLL 계산...")
        normal_nll = calculate_nll_scores(model, normal_dataloader, device)
        
        print(f"  이상 프레임 NLL 계산...")
        abnormal_nll = calculate_nll_scores(model, abnormal_dataloader, device)
        
        # 5. 통계 계산
        normal_mean = np.mean(normal_nll)
        normal_std = np.std(normal_nll)
        abnormal_mean = np.mean(abnormal_nll)
        abnormal_std = np.std(abnormal_nll)
        
        diff = abnormal_mean - normal_mean
        diff_ratio = (diff / abs(normal_mean)) * 100 if normal_mean != 0 else 0
        
        # 6. 결과 출력
        print(f"\n[{part_name}] 검증 결과 (프레임별 NLL):")
        print(f"  {'='*70}")
        print(f"  정상 프레임 세그먼트:   {len(normal_nll)}개, 평균 NLL = {normal_mean:8.4f} ± {normal_std:8.4f}")
        print(f"  이상 프레임 세그먼트:   {len(abnormal_nll)}개, 평균 NLL = {abnormal_mean:8.4f} ± {abnormal_std:8.4f}")
        print(f"  {'='*70}")
        print(f"  차이 (Abnormal - Normal): {diff:8.4f} ({diff_ratio:+.2f}%)")
        print(f"  {'='*70}")
        
        # 판정
        if diff > 0 and diff_ratio > 10:  # 이상 NLL이 정상보다 10% 이상 높음
            status = "✅ 성공"
            message = "STG-NF 모델이 정상/이상을 잘 구분합니다."
        elif diff > 0 and diff_ratio > 5:
            status = "⚠️  경고"
            message = "구분력이 약합니다. 더 긴 학습이 필요할 수 있습니다."
        else:
            status = "❌ 실패"
            message = "STG-NF 모델이 정상/이상을 구분하지 못합니다. 재학습 필요!"
        
        print(f"  판정: {status}")
        print(f"  {message}\n")
        
        # 결과 저장
        results[part_name] = {
            'normal_count': len(normal_nll),
            'abnormal_count': len(abnormal_nll),
            'normal_mean': float(normal_mean),
            'normal_std': float(normal_std),
            'abnormal_mean': float(abnormal_mean),
            'abnormal_std': float(abnormal_std),
            'diff': float(diff),
            'diff_ratio': float(diff_ratio),
            'status': status,
            'message': message
        }
    
    # 최종 요약
    print_section("검증 요약")
    print(f"\n{'부위':<10} {'정상 NLL':>12} {'이상 NLL':>12} {'차이':>10} {'비율':>10} {'상태':>10}")
    print("=" * 80)
    
    for part_name, result in results.items():
        print(f"{part_name:<10} "
              f"{result['normal_mean']:>12.4f} "
              f"{result['abnormal_mean']:>12.4f} "
              f"{result['diff']:>10.4f} "
              f"{result['diff_ratio']:>9.2f}% "
              f"{result['status']:>10}")
    
    print("=" * 80)
    
    # 전체 판정
    failed_parts = [part for part, result in results.items() 
                    if '❌' in result['status']]
    warning_parts = [part for part, result in results.items() 
                     if '⚠️' in result['status']]
    
    print(f"\n전체 결과:")
    if failed_parts:
        print(f"  ❌ 실패한 부위: {', '.join(failed_parts)}")
        print(f"  → STGNFConfig.EPOCHS를 30에서 50~60으로 늘려서 재학습하세요!")
    elif warning_parts:
        print(f"  ⚠️  주의 필요 부위: {', '.join(warning_parts)}")
        print(f"  → 성능 향상을 위해 추가 학습을 고려하세요.")
    else:
        print(f"  ✅ 모든 부위가 정상적으로 학습되었습니다!")
        print(f"  → Attention 모델 학습을 진행할 수 있습니다.")
    
    print("\n")
    
    return results


def main():
    """메인 함수"""
    try:
        results = verify_stgnf_models()
        
        # 결과 JSON 저장 (선택사항)
        import json
        output_dir = os.path.join(Config.Path.RESULTS_DIR, 'stgnf_verification')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'verification_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"결과 저장: {output_path}")
        
    except Exception as e:
        print(f"\n[ERROR] 검증 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
