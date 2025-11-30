"""
STG-NF 모델 검증 스크립트
seg_len=24로 학습된 STG-NF 모델들이 정상/이상 데이터를 제대로 구분하는지 검증

프레임별 NLL 계산 방식 (STG-NF_AI-HUB의 eval.py 방식):
1. 각 세그먼트(24프레임)에 대해 1개의 NLL 값을 계산
2. 메타데이터의 프레임 인덱스 정보를 사용해 해당 NLL을 관련 프레임들에 할당
   예: 0~23프레임 세그먼트 → NLL=0.8 → 이 점수를 0~23 프레임에 모두 할당
3. 겹치는 세그먼트로 인해 한 프레임에 여러 점수가 있을 수 있음 → 평균 사용
4. 최종적으로 각 프레임마다 하나의 NLL 점수 할당
"""

import os
import sys
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from collections import defaultdict

# 설정 import
from args import Config
from datasets.dataset import ShopliftingDataset
from models.model_builder import create_stgnf_model
from datasets.train_utils import set_seed, print_section

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_top_k_mean(scores, k_percent=10):
    """
    점수 배열에서 상위 k%에 해당하는 값들의 평균을 반환합니다.
    도난 행동은 짧은 순간에 발생하므로 전체 평균보다 이 방식이 훨씬 정확합니다.
    """
    if len(scores) == 0:
        return 0.0
    
    # 내림차순 정렬
    sorted_scores = np.sort(scores)[::-1]
    
    # 상위 k% 개수 계산 (최소 1개는 선택)
    num_k = int(len(scores) * (k_percent / 100))
    num_k = max(num_k, 1)
    
    # 상위 k개만 잘라서 평균 계산
    top_k_scores = sorted_scores[:num_k]
    
    return np.mean(top_k_scores)

def load_stgnf_model(part_name, joint_subset, device):
    """부위별 STG-NF 모델 로드"""
    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, f'stgnf_{part_name}_fin.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: 모델 파일 없음: {checkpoint_path}")
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
    
    # ActNorm 통계를 포함한 모든 파라미터를 strict=True로 로드
    # 절대로 ActNorm을 재초기화하지 않음 (training 때의 통계를 그대로 사용해야 함)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    # ActNorm의 inited 플래그는 state_dict에 저장되지 않으므로 수동으로 설정
    # checkpoint에 bias, logs가 있으면 이미 초기화된 것으로 간주
    for name, module in model.named_modules():
        if hasattr(module, 'inited') and hasattr(module, 'bias') and hasattr(module, 'logs'):
            module.inited = True
    
    model.eval()
    
    print(f"  [{part_name}] 모델 로드 완료 (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    if 'discrimination' in checkpoint:
        print(f"  [{part_name}] Discrimination: {checkpoint['discrimination']:.4f}%")
    
    return model

def calculate_frame_level_nll(model, dataset, device, seg_len, seg_stride):
    # 비디오별 프레임별 NLL 점수 저장
    video_frame_scores = defaultdict(lambda: defaultdict(list))
    
    # dataset의 segment_metadata 사용 (이미 생성됨)
    segment_metadata = dataset.segment_metadata
    
    print(f"  총 세그먼트 수: {len(segment_metadata)}")
    
    # 세그먼트별 NLL 계산
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    
    segment_idx = 0
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="  Calculating NLL", leave=False):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # [DEBUG] 데이터 shape 확인
            if segment_idx == 0:
                print(f"    [DEBUG] 데이터 shape: {data.shape}")  # (B, C, T, V)
            
            # Forward pass: 세그먼트별 NLL
            z, nll = model(data)  # nll shape: (B,)
            nll_scores = nll.cpu().numpy()
            
            # [DEBUG] 첫 배치에서 NLL 범위 확인
            if segment_idx == 0:
                print(f"    [DEBUG] 첫 배치 NLL 범위: {nll_scores.min():.4f} ~ {nll_scores.max():.4f}, 평균: {nll_scores.mean():.4f}")
            
            # 각 세그먼트의 NLL을 관련 프레임들에 할당
            for i in range(batch_size):
                if segment_idx >= len(segment_metadata):
                    break
                
                meta = segment_metadata[segment_idx]
                
                # 파일명에서 video_id 추출
                filename = meta['filename']
                video_id = filename.replace('.mp4', '').replace('.json', '')
                
                # start_frame과 seg_len으로 frame_ids 계산
                start_frame = meta['start_frame']
                frame_ids = list(range(start_frame, start_frame + seg_len))
                
                nll_score = float(nll_scores[i])
                
                # 이 세그먼트의 NLL을 모든 관련 프레임에 할당
                for frame_id in frame_ids:
                    video_frame_scores[video_id][frame_id].append(nll_score)
                
                segment_idx += 1
    
    # 각 프레임의 NLL 점수를 max로 집계 (train과 동일하게)
    final_scores = {}
    for video_id, frame_dict in video_frame_scores.items():
        final_scores[video_id] = {}
        for frame_id, scores in frame_dict.items():
            final_scores[video_id][frame_id] = max(scores)  # train과 동일하게 max 사용
    
    return final_scores, segment_metadata

def verify_stgnf_models():
    """STG-NF 모델들 검증"""
    print_section("STG-NF 모델 검증 (프레임별 NLL)")
    
    seg_len_for_stgnf = Config.Data.SEG_LEN
    eval_stride = Config.Data.EVAL_STRIDE
    
    print(f"SEG_LEN: {seg_len_for_stgnf}")
    print(f"EVAL_STRIDE: {eval_stride}")
    print(f"검증할 부위: {Config.Joint.BODY_PARTS}\n")
    
    # 시드 설정
    set_seed(Config.Train.SEED)
    
    # 디바이스 설정
    device = torch.device(Config.Train.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}\n")
    
    # GT 데이터 로드 (프레임별 라벨)
    print("GT 데이터 로드 중...")
    
    # GT 폴더에서 직접 .npy 파일들을 스캔
    video_frame_gt = {}
    gt_files = glob.glob(os.path.join(Config.Path.TEST_GT_DIR, '*.npy'))
    
    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        video_id = filename.replace('.npy', '')
        
        # numpy 배열 로드 (frame_idx: 0 or 1)
        gt_array = np.load(gt_path)
        
        # 프레임별 GT (0=normal, 1=abnormal)
        frame_labels = {i: int(gt_array[i]) for i in range(len(gt_array))}
        video_frame_gt[video_id] = frame_labels
    
    print(f"GT 비디오 수: {len(video_frame_gt)}\n")
    
    # 결과 저장용
    results = {}
    
    # 부위별 검증
    for part_name in Config.Joint.BODY_PARTS:
        print_section(f"[{part_name.upper()}] 부위 검증")
        
        joint_subset = Config.Joint.JOINT_SUBSET_MAP[part_name]
        
        # 1. 테스트 데이터셋 먼저 로드 (ActNorm 초기화에 사용)
        print(f"\n[{part_name}] 테스트 데이터 로드 중...")
        test_dataset = ShopliftingDataset(
            skeleton_base_path=Config.Path.TEST_DATA_DIR,
            seg_len=Config.Data.SEG_LEN,
            seg_stride=1,  # train과 동일하게 명시적으로 1
            joint_subset=joint_subset,
            normalize=Config.Data.NORMALIZE,
            apply_augmentation=False,
            vid_res=Config.Data.VID_RES,
            use_cache=Config.Data.USE_CACHE,
            load_per_batch=False,
            preprocess_cache=True
        )
        
        print(f"  총 세그먼트 수: {len(test_dataset)}")
        print(f"  정상 세그먼트: {sum([1 for l in test_dataset.segment_labels if l == 0])}")
        print(f"  이상 세그먼트: {sum([1 for l in test_dataset.segment_labels if l == 1])}")
        
        # 2. STG-NF 모델 로드 (실제 데이터로 ActNorm 초기화)
        print(f"\n[{part_name}] STG-NF 모델 로드 중...")
        model = load_stgnf_model(part_name, joint_subset, device)
        
        if model is None:
            print(f"[{part_name}] 모델 로드 실패. 스킵.\n")
            continue
        
        # 3. 프레임별 NLL 계산
        print(f"\n[{part_name}] 프레임별 NLL 계산 중...")
        video_frame_nll, segment_metadata = calculate_frame_level_nll(
            model, test_dataset, device, seg_len_for_stgnf, eval_stride
        )
        
        # 4. 프레임별 정상/이상 분리 및 통계 계산
        normal_frame_nlls = []
        abnormal_frame_nlls = []
        
        for video_id, frame_dict in video_frame_nll.items():
            if video_id not in video_frame_gt:
                continue
            
            gt_frames = video_frame_gt[video_id]
            
            for frame_id, nll_score in frame_dict.items():
                if frame_id not in gt_frames:
                    continue
                
                frame_label = gt_frames[frame_id]
                
                if frame_label == 0:  # Normal
                    normal_frame_nlls.append(nll_score)
                else:  # Abnormal
                    abnormal_frame_nlls.append(nll_score)
        
        if len(normal_frame_nlls) == 0 or len(abnormal_frame_nlls) == 0:
            print(f"[{part_name}] 데이터 부족. 스킵.\n")
            continue
        
        normal_nll = np.array(normal_frame_nlls)
        abnormal_nll = np.array(abnormal_frame_nlls)
        
        # ===== 전체 평균 비교 (기본 평가) =====
        normal_mean = np.mean(normal_nll)
        abnormal_mean = np.mean(abnormal_nll)
        
        normal_std = np.std(normal_nll)
        abnormal_std = np.std(abnormal_nll)
        
        diff = abnormal_mean - normal_mean
        diff_ratio = (diff / abs(normal_mean)) * 100 if normal_mean != 0 else 0
        
        # ===== 추가 통계: Top-10% 참고용 =====
        normal_top10 = get_top_k_mean(normal_nll, k_percent=10)
        abnormal_top10 = get_top_k_mean(abnormal_nll, k_percent=10)
        
        # 결과 출력
        print(f"\n[{part_name}] 검증 결과 (전체 평균 기준):")
        print(f"  {'='*70}")
        print(f"  정상 프레임:   Mean={normal_mean:8.4f}, Std={normal_std:.4f}, Top10%={normal_top10:.4f}")
        print(f"  이상 프레임:   Mean={abnormal_mean:8.4f}, Std={abnormal_std:.4f}, Top10%={abnormal_top10:.4f}")
        print(f"  {'='*70}")
        print(f"  차이 (Abnormal - Normal): {diff:8.4f} ({diff_ratio:+.2f}%)")
        print(f"  {'='*70}")
        
        # 판정 (더 현실적인 기준)
        if diff > 0 and diff_ratio > 3:
            status = "✅ 성공"
            message = "STG-NF 모델이 정상/이상을 구분합니다. Attention 학습 가능."
        elif diff > 0 and diff_ratio > 1:
            status = "⚠️  경고"
            message = "구분력이 약합니다. Attention이 증폭할 수 있지만 재학습 고려."
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
        print(f"  → STGNFConfig.EPOCHS를 늘려서 재학습하세요!")
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
        print(f"\nError: 검증 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
