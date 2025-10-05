"""
JSON 키포인트 데이터를 HDF5 형식으로 변환하는 스크립트

데이터 구조:
- JSON: 사람ID -> 프레임 -> 키포인트 좌표 (x, y, confidence)
- HDF5: 효율적인 저장과 빠른 액세스를 위한 계층적 데이터 형식

사용법:
python json_to_hdf5_converter.py
"""

import json
import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse


def parse_keypoints(keypoints_list):
    """키포인트 리스트를 numpy 배열로 변환
    
    Args:
        keypoints_list: [x1, y1, conf1, x2, y2, conf2, ...] 형태의 리스트
        
    Returns:
        numpy array: shape (num_keypoints, 3) - [x, y, confidence]
    """
    if keypoints_list is None or len(keypoints_list) == 0:
        return np.array([])
    
    keypoints_array = np.array(keypoints_list)
    # 3개씩 묶어서 (x, y, confidence) 형태로 변환
    num_keypoints = len(keypoints_array) // 3
    return keypoints_array.reshape(num_keypoints, 3)


def load_ground_truth(gt_file_path):
    """Ground truth 파일 로드 (.npy 파일)
    
    Args:
        gt_file_path: ground truth 파일 경로
        
    Returns:
        numpy array: ground truth 레이블
    """
    if os.path.exists(gt_file_path):
        return np.load(gt_file_path)
    else:
        print(f"Warning: Ground truth file not found: {gt_file_path}")
        return None


def convert_json_to_hdf5(json_file_path, output_dir, gt_dir=None):
    """단일 JSON 파일을 HDF5로 변환
    
    Args:
        json_file_path: 입력 JSON 파일 경로
        output_dir: 출력 디렉토리
        gt_dir: ground truth 디렉토리 (있는 경우)
    """
    
    # JSON 파일 로드
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 출력 파일명 생성
    json_filename = Path(json_file_path).stem
    hdf5_filename = f"{json_filename}.h5"
    output_path = os.path.join(output_dir, hdf5_filename)
    
    # Ground truth 파일 경로
    gt_labels = None
    if gt_dir:
        gt_file_path = os.path.join(gt_dir, f"{json_filename}_gt.npy")
        gt_labels = load_ground_truth(gt_file_path)
    
    # HDF5 파일 생성
    with h5py.File(output_path, 'w') as hf:
        # 메타데이터 저장
        hf.attrs['source_file'] = json_filename
        hf.attrs['total_persons'] = len(data)
        
        # Ground truth가 있으면 저장
        if gt_labels is not None:
            hf.create_dataset('ground_truth', data=gt_labels, compression='gzip')
            hf.attrs['has_ground_truth'] = True
        else:
            hf.attrs['has_ground_truth'] = False
        
        # 각 사람별로 데이터 처리
        for person_id, person_data in data.items():
            # 사람 그룹 생성
            person_group = hf.create_group(person_id)
            person_group.attrs['total_frames'] = len(person_data)
            
            # 프레임별 키포인트 저장을 위한 리스트
            frame_numbers = []
            keypoints_data = []
            scores_data = []
            
            # 각 프레임의 데이터 처리
            for frame_id, frame_data in person_data.items():
                frame_numbers.append(int(frame_id))
                
                # 키포인트 파싱
                keypoints = parse_keypoints(frame_data.get('keypoints', []))
                scores = frame_data.get('scores', None)
                
                keypoints_data.append(keypoints)
                if scores is not None:
                    scores_data.append(np.array(scores))
                else:
                    scores_data.append(None)
            
            # 프레임 번호 순으로 정렬
            sorted_indices = np.argsort(frame_numbers)
            sorted_frames = [frame_numbers[i] for i in sorted_indices]
            sorted_keypoints = [keypoints_data[i] for i in sorted_indices]
            sorted_scores = [scores_data[i] for i in sorted_indices]
            
            # 프레임 번호 저장
            person_group.create_dataset('frame_numbers', data=np.array(sorted_frames), compression='gzip')
            
            # 키포인트 데이터 저장
            # 모든 프레임의 키포인트 크기가 같은지 확인
            keypoint_shapes = [kp.shape for kp in sorted_keypoints if len(kp) > 0]
            if keypoint_shapes:
                # 가장 일반적인 키포인트 수 찾기
                most_common_shape = max(set(keypoint_shapes), key=keypoint_shapes.count)
                num_keypoints = most_common_shape[0]
                
                # 모든 프레임을 같은 크기로 맞춤 (패딩 또는 자르기)
                normalized_keypoints = []
                for kp in sorted_keypoints:
                    if len(kp) == 0:
                        # 빈 키포인트는 0으로 채움
                        normalized_kp = np.zeros((num_keypoints, 3))
                    elif kp.shape[0] < num_keypoints:
                        # 부족한 키포인트는 0으로 패딩
                        padding = np.zeros((num_keypoints - kp.shape[0], 3))
                        normalized_kp = np.vstack([kp, padding])
                    elif kp.shape[0] > num_keypoints:
                        # 초과하는 키포인트는 자름
                        normalized_kp = kp[:num_keypoints]
                    else:
                        normalized_kp = kp
                    
                    normalized_keypoints.append(normalized_kp)
                
                # 3D 배열로 변환: (frames, keypoints, 3)
                keypoints_array = np.stack(normalized_keypoints, axis=0)
                person_group.create_dataset('keypoints', data=keypoints_array, compression='gzip')
                person_group.attrs['num_keypoints'] = num_keypoints
            
            # 스코어 데이터 저장 (있는 경우)
            valid_scores = [s for s in sorted_scores if s is not None]
            if valid_scores:
                # 스코어도 같은 방식으로 정규화
                max_score_len = max(len(s) for s in valid_scores)
                normalized_scores = []
                
                for s in sorted_scores:
                    if s is None:
                        normalized_s = np.zeros(max_score_len)
                    elif len(s) < max_score_len:
                        padding = np.zeros(max_score_len - len(s))
                        normalized_s = np.concatenate([s, padding])
                    elif len(s) > max_score_len:
                        normalized_s = s[:max_score_len]
                    else:
                        normalized_s = s
                    
                    normalized_scores.append(normalized_s)
                
                scores_array = np.stack(normalized_scores, axis=0)
                person_group.create_dataset('scores', data=scores_array, compression='gzip')
    
    return output_path


def convert_dataset(input_dir, output_dir, gt_dir=None):
    """전체 데이터셋을 JSON에서 HDF5로 변환
    
    Args:
        input_dir: JSON 파일들이 있는 디렉토리
        output_dir: HDF5 파일들을 저장할 디렉토리
        gt_dir: Ground truth 파일들이 있는 디렉토리 (선택사항)
    """
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 파일 목록 가져오기
    json_files = list(Path(input_dir).glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    print(f"Converting to HDF5 format...")
    
    successful_conversions = 0
    failed_conversions = 0
    
    # 각 JSON 파일 변환
    for json_file in tqdm(json_files, desc="Converting files"):
        try:
            output_path = convert_json_to_hdf5(json_file, output_dir, gt_dir)
            successful_conversions += 1
            # print(f"✓ Converted: {json_file.name} -> {Path(output_path).name}")
        except Exception as e:
            print(f"✗ Failed to convert {json_file.name}: {str(e)}")
            failed_conversions += 1
    
    print(f"\nConversion completed!")
    print(f"Successful: {successful_conversions}")
    print(f"Failed: {failed_conversions}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert JSON keypoint data to HDF5 format')
    parser.add_argument('--input_dir', type=str, default='dataset_output',
                      help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default='dataset_hdf5',
                      help='Directory to save HDF5 files')
    parser.add_argument('--include_gt', action='store_true',
                      help='Include ground truth data if available')
    
    args = parser.parse_args()
    
    # 경로 설정
    script_dir = Path(__file__).parent
    input_base_dir = script_dir / args.input_dir
    output_base_dir = script_dir / args.output_dir
    
    # Train 데이터 변환
    train_input_dir = input_base_dir / "train"
    train_output_dir = output_base_dir / "train"
    
    if train_input_dir.exists():
        print("=== Converting Training Data ===")
        convert_dataset(train_input_dir, train_output_dir)
    
    # Test 데이터 변환
    test_input_dir = input_base_dir / "test"
    test_output_dir = output_base_dir / "test"
    
    if test_input_dir.exists():
        print("\n=== Converting Test Data ===")
        gt_dir = None
        if args.include_gt:
            gt_dir = test_input_dir / "ground_truth"
            if not gt_dir.exists():
                print("Warning: Ground truth directory not found, proceeding without GT data")
                gt_dir = None
        
        convert_dataset(test_input_dir, test_output_dir, gt_dir)
    
    print(f"\n=== Conversion Summary ===")
    print(f"Input directory: {input_base_dir}")
    print(f"Output directory: {output_base_dir}")
    
    # HDF5 파일 통계
    total_hdf5_files = len(list(output_base_dir.rglob("*.h5")))
    print(f"Total HDF5 files created: {total_hdf5_files}")


if __name__ == "__main__":
    main()