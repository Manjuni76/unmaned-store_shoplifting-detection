"""
XML 파일의 theft_start/theft_end 정보를 기반으로 
HDF5와 NPY 파일의 ground truth를 정확하게 수정
"""

import h5py
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import json


def find_xml_file(filename_base, xml_dirs):
    """파일명에 해당하는 XML 파일 찾기"""
    # filename_base에서 abnormal_shoplifting_XXX_ 제거
    if filename_base.startswith('abnormal_shoplifting_'):
        # abnormal_shoplifting_001_C_3_12_16_BU_SMB_09-01_14-40-24_CD_RGB_DF2_F2
        # -> C_3_12_16_BU_SMB_09-01_14-40-24_CD_RGB_DF2_F2
        parts = filename_base.split('_')
        if len(parts) >= 4 and parts[0] == 'abnormal' and parts[1] == 'shoplifting':
            # abnormal_shoplifting_001_ 부분 제거하고 나머지 조합
            xml_base = '_'.join(parts[3:])  # C_3_12_16_BU_SMB_09-01_14-40-24_CD_RGB_DF2_F2
            xml_pattern = f"{xml_base}.xml"
            
            for xml_dir in xml_dirs:
                xml_files = list(xml_dir.glob(f"**/{xml_pattern}"))
                if xml_files:
                    return xml_files[0]
    
    return None


def extract_theft_frames(xml_file):
    """XML 파일에서 theft_start와 theft_end 프레임 추출"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        theft_start_frames = []
        theft_end_frames = []
        
        # track 요소들 찾기
        for track in root.findall('.//track'):
            label = track.get('label', '')
            if label == 'theft_start':
                boxes = track.findall('box')
                if boxes:
                    frames = [int(box.get('frame', 0)) for box in boxes]
                    theft_start_frames.extend(frames)
            elif label == 'theft_end':
                boxes = track.findall('box')
                if boxes:
                    frames = [int(box.get('frame', 0)) for box in boxes]
                    theft_end_frames.extend(frames)
        
        if theft_start_frames and theft_end_frames:
            start_frame = min(theft_start_frames)
            end_frame = max(theft_end_frames)
            return start_frame, end_frame
        
        return None, None
        
    except Exception as e:
        print(f"XML 파싱 오류 - {xml_file}: {e}")
        return None, None


def create_correct_ground_truth(total_frames, start_frame, end_frame):
    """올바른 ground truth 배열 생성"""
    gt_array = np.zeros(total_frames, dtype=np.int32)
    
    if start_frame is not None and end_frame is not None:
        # theft 구간을 1로 설정
        gt_array[start_frame:end_frame+1] = 1
    
    return gt_array


def fix_hdf5_ground_truth():
    """HDF5 파일들의 ground truth 수정"""
    
    print("🔧 HDF5 파일 Ground Truth 수정 중...")
    
    # XML 디렉토리들
    xml_dirs = [
        Path(r"D:\AI-HUB_shoplifting\shoplift_data\Training\label_data\Shoplift"),
        Path(r"D:\AI-HUB_shoplifting\shoplift_data\Validation\label_data\Shoplift")
    ]
    
    # HDF5 디렉토리들
    hdf5_dirs = [
        Path("dataset_hdf5/train"),
        Path("dataset_hdf5/test")
    ]
    
    for hdf5_dir in hdf5_dirs:
        if not hdf5_dir.exists():
            continue
            
        print(f"\n📁 처리 중: {hdf5_dir}")
        
        # abnormal 파일들만 처리
        abnormal_files = [f for f in hdf5_dir.glob("abnormal_*.h5")]
        
        success_count = 0
        fail_count = 0
        
        for h5_file in tqdm(abnormal_files, desc=f"{hdf5_dir.name} 수정"):
            filename_base = h5_file.stem
            
            # 해당하는 XML 파일 찾기
            xml_file = find_xml_file(filename_base, xml_dirs)
            
            if xml_file is None:
                print(f"❌ XML 파일 없음: {filename_base}")
                fail_count += 1
                continue
            
            # XML에서 theft 구간 추출
            start_frame, end_frame = extract_theft_frames(xml_file)
            
            if start_frame is None or end_frame is None:
                print(f"❌ theft 구간 없음: {filename_base}")
                fail_count += 1
                continue
            
            try:
                # HDF5 파일 수정
                with h5py.File(h5_file, 'r+') as f:
                    if 'ground_truth' in f:
                        current_gt = f['ground_truth'][:]
                        total_frames = len(current_gt)
                        
                        # 새로운 ground truth 생성
                        new_gt = create_correct_ground_truth(total_frames, start_frame, end_frame)
                        
                        # 업데이트
                        del f['ground_truth']
                        f.create_dataset('ground_truth', data=new_gt, compression='gzip')
                        
                        print(f"✅ {filename_base}: 프레임 {start_frame}~{end_frame} → 라벨=1")
                        success_count += 1
                    else:
                        print(f"❌ ground_truth 없음: {filename_base}")
                        fail_count += 1
                        
            except Exception as e:
                print(f"❌ HDF5 수정 오류 - {filename_base}: {e}")
                fail_count += 1
        
        print(f"\n📊 {hdf5_dir.name} 결과:")
        print(f"  성공: {success_count}개")
        print(f"  실패: {fail_count}개")


def fix_npy_ground_truth():
    """NPY 파일들의 ground truth 수정"""
    
    print("\n🔧 NPY 파일 Ground Truth 수정 중...")
    
    # XML 디렉토리들
    xml_dirs = [
        Path(r"D:\AI-HUB_shoplifting\shoplift_data\Training\label_data\Shoplift"),
        Path(r"D:\AI-HUB_shoplifting\shoplift_data\Validation\label_data\Shoplift")
    ]
    
    # NPY 디렉토리들
    npy_dirs = [
        Path("dataset_output/train/ground_truth"),
        Path("dataset_output/test/ground_truth")
    ]
    
    for npy_dir in npy_dirs:
        if not npy_dir.exists():
            continue
            
        print(f"\n📁 처리 중: {npy_dir}")
        
        # abnormal 파일들만 처리
        abnormal_files = [f for f in npy_dir.glob("abnormal_*_gt.npy")]
        
        success_count = 0
        fail_count = 0
        
        for npy_file in tqdm(abnormal_files, desc=f"{npy_dir.parent.name} 수정"):
            filename_base = npy_file.stem.replace('_gt', '')  # _gt.npy 제거
            
            # 해당하는 XML 파일 찾기
            xml_file = find_xml_file(filename_base, xml_dirs)
            
            if xml_file is None:
                print(f"❌ XML 파일 없음: {filename_base}")
                fail_count += 1
                continue
            
            # XML에서 theft 구간 추출
            start_frame, end_frame = extract_theft_frames(xml_file)
            
            if start_frame is None or end_frame is None:
                print(f"❌ theft 구간 없음: {filename_base}")
                fail_count += 1
                continue
            
            try:
                # 현재 NPY 파일 로드
                current_gt = np.load(npy_file)
                total_frames = len(current_gt)
                
                # 새로운 ground truth 생성
                new_gt = create_correct_ground_truth(total_frames, start_frame, end_frame)
                
                # 저장
                np.save(npy_file, new_gt)
                
                print(f"✅ {filename_base}: 프레임 {start_frame}~{end_frame} → 라벨=1")
                success_count += 1
                        
            except Exception as e:
                print(f"❌ NPY 수정 오류 - {filename_base}: {e}")
                fail_count += 1
        
        print(f"\n📊 {npy_dir.parent.name} 결과:")
        print(f"  성공: {success_count}개")
        print(f"  실패: {fail_count}개")


def verify_corrections():
    """수정 결과 검증"""
    
    print("\n🔍 수정 결과 검증 중...")
    
    # 테스트 파일 몇 개 확인
    test_files = [
        "abnormal_shoplifting_001_C_3_12_16_BU_SMB_09-01_14-40-24_CD_RGB_DF2_F2",
        "abnormal_shoplifting_002_C_3_12_41_BU_SMC_10-14_11-45-31_CD_RGB_DF2_M2"
    ]
    
    for filename in test_files:
        print(f"\n📋 {filename} 검증:")
        
        # HDF5 확인
        hdf5_file = Path(f"dataset_hdf5/test/{filename}.h5")
        if hdf5_file.exists():
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    if 'ground_truth' in f:
                        gt_data = f['ground_truth'][:]
                        unique, counts = np.unique(gt_data, return_counts=True)
                        label_dist = dict(zip(unique, counts))
                        
                        # 라벨=1인 구간 찾기
                        abnormal_indices = np.where(gt_data == 1)[0]
                        if len(abnormal_indices) > 0:
                            start_idx = abnormal_indices[0]
                            end_idx = abnormal_indices[-1]
                            print(f"  HDF5: 이상 구간 {start_idx}~{end_idx}, 분포 {label_dist}")
                        else:
                            print(f"  HDF5: 이상 구간 없음, 분포 {label_dist}")
            except Exception as e:
                print(f"  HDF5 확인 오류: {e}")
        
        # NPY 확인
        npy_file = Path(f"dataset_output/test/ground_truth/{filename}_gt.npy")
        if npy_file.exists():
            try:
                gt_data = np.load(npy_file)
                unique, counts = np.unique(gt_data, return_counts=True)
                label_dist = dict(zip(unique, counts))
                
                # 라벨=1인 구간 찾기
                abnormal_indices = np.where(gt_data == 1)[0]
                if len(abnormal_indices) > 0:
                    start_idx = abnormal_indices[0]
                    end_idx = abnormal_indices[-1]
                    print(f"  NPY:  이상 구간 {start_idx}~{end_idx}, 분포 {label_dist}")
                else:
                    print(f"  NPY:  이상 구간 없음, 분포 {label_dist}")
            except Exception as e:
                print(f"  NPY 확인 오류: {e}")


def main():
    print("🚀 XML 기반 Ground Truth 수정 시작")
    print("=" * 60)
    
    # 1. HDF5 파일 수정
    fix_hdf5_ground_truth()
    
    # 2. NPY 파일 수정
    fix_npy_ground_truth()
    
    # 3. 결과 검증
    verify_corrections()
    
    print("\n🎉 XML 기반 Ground Truth 수정 완료!")
    print("이제 abnormal 파일들이 XML의 theft_start~theft_end 구간만 라벨=1로 설정됩니다.")


if __name__ == "__main__":
    main()