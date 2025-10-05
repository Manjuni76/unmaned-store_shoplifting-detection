"""
HDF5 파일에서 Ground Truth를 추출하여 별도 폴더에 저장
파일명이 매치되도록 구성
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_ground_truth_from_hdf5():
    """HDF5 파일에서 ground truth를 추출하여 별도 폴더에 저장"""
    
    print("🔄 HDF5에서 Ground Truth 추출 중...")
    
    # 입력/출력 디렉토리
    hdf5_dirs = [
        Path("dataset_hdf5/train"),
        Path("dataset_hdf5/test")
    ]
    
    for hdf5_dir in hdf5_dirs:
        if not hdf5_dir.exists():
            continue
            
        print(f"\n📁 처리 중: {hdf5_dir}")
        
        # Ground truth 출력 디렉토리 생성
        gt_output_dir = Path(f"dataset_hdf5_ground_truth/{hdf5_dir.name}")
        gt_output_dir.mkdir(parents=True, exist_ok=True)
        
        # HDF5 파일들 처리
        h5_files = [f for f in hdf5_dir.glob("*.h5") if f.stem != "video_metadata"]
        
        success_count = 0
        fail_count = 0
        
        for h5_file in tqdm(h5_files, desc=f"{hdf5_dir.name} 추출"):
            try:
                filename_base = h5_file.stem  # 확장자 제거
                
                with h5py.File(h5_file, 'r') as f:
                    if 'ground_truth' in f:
                        gt_data = f['ground_truth'][:]
                        
                        # Ground truth 파일 저장
                        gt_filename = f"{filename_base}_gt.npy"
                        gt_filepath = gt_output_dir / gt_filename
                        
                        np.save(gt_filepath, gt_data)
                        
                        success_count += 1
                        
                        # 간단한 검증
                        unique, counts = np.unique(gt_data, return_counts=True)
                        label_dist = dict(zip(unique, counts))
                        
                        print(f"✅ {filename_base}: {len(gt_data)}프레임, 분포 {label_dist}")
                        
                    else:
                        print(f"❌ Ground truth 없음: {filename_base}")
                        fail_count += 1
                        
            except Exception as e:
                print(f"❌ 처리 오류 - {h5_file.name}: {e}")
                fail_count += 1
        
        print(f"\n📊 {hdf5_dir.name} 결과:")
        print(f"  성공: {success_count}개")
        print(f"  실패: {fail_count}개")
        print(f"  저장 위치: {gt_output_dir}")


def verify_extracted_ground_truth():
    """추출된 ground truth 파일들 검증"""
    
    print("\n🔍 추출된 Ground Truth 파일 검증...")
    
    gt_dirs = [
        Path("dataset_hdf5_ground_truth/train"),
        Path("dataset_hdf5_ground_truth/test")
    ]
    
    for gt_dir in gt_dirs:
        if not gt_dir.exists():
            continue
            
        print(f"\n📁 검증 중: {gt_dir}")
        
        gt_files = list(gt_dir.glob("*_gt.npy"))
        
        total_normal_frames = 0
        total_abnormal_frames = 0
        normal_files = 0
        abnormal_files = 0
        
        for gt_file in sorted(gt_files):
            filename_base = gt_file.stem.replace('_gt', '')
            is_abnormal = filename_base.startswith('abnormal_')
            
            try:
                gt_data = np.load(gt_file)
                unique, counts = np.unique(gt_data, return_counts=True)
                label_counts = dict(zip(unique, counts))
                
                zeros = label_counts.get(0, 0)
                ones = label_counts.get(1, 0)
                
                total_normal_frames += zeros
                total_abnormal_frames += ones
                
                if is_abnormal:
                    abnormal_files += 1
                else:
                    normal_files += 1
                    
            except Exception as e:
                print(f"❌ 검증 오류 - {gt_file.name}: {e}")
        
        print(f"  총 파일: {len(gt_files)}개")
        print(f"  정상 파일: {normal_files}개")
        print(f"  이상 파일: {abnormal_files}개")
        print(f"  정상 프레임 (라벨=0): {total_normal_frames:,}개")
        print(f"  이상 프레임 (라벨=1): {total_abnormal_frames:,}개")
        
        if total_normal_frames + total_abnormal_frames > 0:
            normal_ratio = total_normal_frames / (total_normal_frames + total_abnormal_frames) * 100
            abnormal_ratio = total_abnormal_frames / (total_normal_frames + total_abnormal_frames) * 100
            print(f"  정상:이상 비율 = {normal_ratio:.1f}%:{abnormal_ratio:.1f}%")


def compare_with_original_npy():
    """원본 NPY 파일과 비교"""
    
    print("\n🔄 원본 NPY와 추출된 Ground Truth 비교...")
    
    # 테스트 데이터만 비교 (원본 NPY가 있는 경우)
    original_npy_dir = Path("dataset_output/test/ground_truth")
    extracted_gt_dir = Path("dataset_hdf5_ground_truth/test")
    
    if not original_npy_dir.exists() or not extracted_gt_dir.exists():
        print("비교할 디렉토리가 없습니다.")
        return
    
    original_files = list(original_npy_dir.glob("*_gt.npy"))
    extracted_files = list(extracted_gt_dir.glob("*_gt.npy"))
    
    print(f"원본 NPY: {len(original_files)}개")
    print(f"추출된 GT: {len(extracted_files)}개")
    
    # 파일명 매칭해서 비교
    match_count = 0
    mismatch_count = 0
    
    for orig_file in original_files:
        extracted_file = extracted_gt_dir / orig_file.name
        
        if extracted_file.exists():
            try:
                orig_data = np.load(orig_file)
                extracted_data = np.load(extracted_file)
                
                if np.array_equal(orig_data, extracted_data):
                    match_count += 1
                else:
                    mismatch_count += 1
                    print(f"❌ 불일치: {orig_file.name}")
                    print(f"  원본: {len(orig_data)}프레임, 추출: {len(extracted_data)}프레임")
                    
            except Exception as e:
                print(f"❌ 비교 오류 - {orig_file.name}: {e}")
                mismatch_count += 1
        else:
            print(f"❌ 추출된 파일 없음: {orig_file.name}")
            mismatch_count += 1
    
    print(f"\n📊 비교 결과:")
    print(f"  일치: {match_count}개")
    print(f"  불일치: {mismatch_count}개")
    
    if match_count > 0 and mismatch_count == 0:
        print("✅ 모든 파일이 완벽하게 일치합니다!")
    elif match_count > 0:
        print("⚠️ 일부 파일에 차이가 있습니다.")
    else:
        print("❌ 모든 파일이 불일치합니다.")


def main():
    print("🚀 HDF5에서 Ground Truth 추출 작업 시작")
    print("=" * 60)
    
    # 1. Ground Truth 추출
    extract_ground_truth_from_hdf5()
    
    # 2. 추출된 파일 검증
    verify_extracted_ground_truth()
    
    # 3. 원본과 비교 (있는 경우)
    compare_with_original_npy()
    
    print("\n🎉 Ground Truth 추출 완료!")
    print("📁 저장 위치:")
    print("  - dataset_hdf5_ground_truth/train/")
    print("  - dataset_hdf5_ground_truth/test/")
    print("\n💡 파일명 형식: [원본파일명]_gt.npy")


if __name__ == "__main__":
    main()