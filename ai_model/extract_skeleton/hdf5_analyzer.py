"""
HDF5 키포인트 데이터를 읽고 분석하는 유틸리티

HDF5 파일의 구조를 탐색하고 데이터를 시각화하는 기능을 제공합니다.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import Counter


def explore_hdf5_structure(file_path):
    """HDF5 파일의 구조를 탐색하고 출력
    
    Args:
        file_path: HDF5 파일 경로
    """
    
    def print_structure(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name}/ (Group)")
            # 속성 출력
            for attr_name, attr_value in obj.attrs.items():
                print(f"{indent}  @{attr_name}: {attr_value}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset): shape={obj.shape}, dtype={obj.dtype}")
            # 속성 출력
            for attr_name, attr_value in obj.attrs.items():
                print(f"{indent}  @{attr_name}: {attr_value}")
    
    print(f"\n=== HDF5 File Structure: {Path(file_path).name} ===")
    
    with h5py.File(file_path, 'r') as f:
        # 파일 레벨 속성
        print("File attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  @{attr_name}: {attr_value}")
        
        print("\nFile structure:")
        f.visititems(print_structure)


def load_keypoints_data(file_path, person_id=None):
    """HDF5 파일에서 키포인트 데이터 로드
    
    Args:
        file_path: HDF5 파일 경로
        person_id: 특정 사람 ID (None이면 모든 사람)
        
    Returns:
        dict: 사람별 키포인트 데이터
    """
    
    data = {}
    
    with h5py.File(file_path, 'r') as f:
        # Ground truth 로드 (있는 경우)
        ground_truth = None
        if 'ground_truth' in f:
            ground_truth = f['ground_truth'][:]
        
        # 각 사람의 데이터 로드
        for person_key in f.keys():
            if person_key == 'ground_truth':
                continue
                
            if person_id is not None and person_key != person_id:
                continue
                
            person_group = f[person_key]
            
            person_data = {
                'frame_numbers': person_group['frame_numbers'][:],
                'keypoints': person_group['keypoints'][:],
                'total_frames': person_group.attrs.get('total_frames', 0),
                'num_keypoints': person_group.attrs.get('num_keypoints', 0)
            }
            
            # 스코어 데이터 (있는 경우)
            if 'scores' in person_group:
                person_data['scores'] = person_group['scores'][:]
            
            data[person_key] = person_data
        
        data['ground_truth'] = ground_truth
    
    return data


def analyze_dataset_statistics(hdf5_dir):
    """데이터셋의 통계 정보 분석
    
    Args:
        hdf5_dir: HDF5 파일들이 있는 디렉토리
    """
    
    hdf5_files = list(Path(hdf5_dir).glob("*.h5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {hdf5_dir}")
        return
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total files: {len(hdf5_files)}")
    
    total_persons = 0
    total_frames = 0
    keypoint_counts = []
    frame_counts = []
    files_with_gt = 0
    
    for hdf5_file in tqdm(hdf5_files, desc="Analyzing files"):
        try:
            with h5py.File(hdf5_file, 'r') as f:
                # 파일별 통계
                file_persons = f.attrs.get('total_persons', 0)
                has_gt = f.attrs.get('has_ground_truth', False)
                
                total_persons += file_persons
                if has_gt:
                    files_with_gt += 1
                
                # 각 사람별 통계
                for person_key in f.keys():
                    if person_key == 'ground_truth':
                        continue
                    
                    person_group = f[person_key]
                    person_frames = person_group.attrs.get('total_frames', 0)
                    person_keypoints = person_group.attrs.get('num_keypoints', 0)
                    
                    total_frames += person_frames
                    frame_counts.append(person_frames)
                    keypoint_counts.append(person_keypoints)
                    
        except Exception as e:
            print(f"Error analyzing {hdf5_file}: {str(e)}")
    
    print(f"Total persons: {total_persons}")
    print(f"Total frames: {total_frames}")
    print(f"Files with ground truth: {files_with_gt}")
    
    if frame_counts:
        print(f"\nFrame statistics per person:")
        print(f"  Mean: {np.mean(frame_counts):.2f}")
        print(f"  Std: {np.std(frame_counts):.2f}")
        print(f"  Min: {np.min(frame_counts)}")
        print(f"  Max: {np.max(frame_counts)}")
    
    if keypoint_counts:
        unique_keypoints = np.unique(keypoint_counts)
        print(f"\nKeypoint counts: {unique_keypoints}")
        for count in unique_keypoints:
            num_persons = np.sum(np.array(keypoint_counts) == count)
            print(f"  {count} keypoints: {num_persons} persons")


def analyze_ground_truth_labels(hdf5_dir):
    """HDF5 데이터셋의 ground truth 라벨 분석
    
    Args:
        hdf5_dir: HDF5 파일들이 있는 디렉토리 경로
    """
    
    hdf5_dir = Path(hdf5_dir)
    h5_files = list(hdf5_dir.glob("*.h5"))
    
    if not h5_files:
        print(f"No HDF5 files found in {hdf5_dir}")
        return
    
    print(f"\n🏷️ Ground Truth 라벨 분석 ({hdf5_dir.name})")
    print("=" * 50)
    
    total_normal_frames = 0
    total_abnormal_frames = 0
    normal_files = 0
    abnormal_files = 0
    files_with_gt = 0
    files_without_gt = 0
    
    label_distribution = []
    file_details = []
    
    # 메타데이터 파일 제외
    h5_files = [f for f in h5_files if f.stem != "video_metadata"]
    
    for h5_file in tqdm(sorted(h5_files), desc="분석 중"):
            
        filename = h5_file.stem
        is_abnormal_file = filename.startswith('abnormal_')
        
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'ground_truth' in f:
                    files_with_gt += 1
                    gt_data = f['ground_truth'][:]
                    
                    # 라벨 개수 세기
                    label_counts = Counter(gt_data)
                    zeros = label_counts.get(0, 0)  # 정상
                    ones = label_counts.get(1, 0)   # 이상
                    
                    total_normal_frames += zeros
                    total_abnormal_frames += ones
                    
                    # 파일 분류
                    if is_abnormal_file:
                        abnormal_files += 1
                    else:
                        normal_files += 1
                    
                    # 상세 정보 저장
                    file_details.append({
                        'filename': filename,
                        'is_abnormal': is_abnormal_file,
                        'normal_frames': zeros,
                        'abnormal_frames': ones,
                        'total_frames': len(gt_data)
                    })
                    
                    # 라벨 분포 저장
                    label_distribution.extend(gt_data)
                    
                else:
                    files_without_gt += 1
                    print(f"⚠️  Ground truth 없음: {filename}")
                    
        except Exception as e:
            print(f"❌ 파일 처리 오류 - {filename}: {str(e)}")
    
    # 결과 출력
    print(f"\n📊 파일 통계:")
    print(f"  총 파일 개수: {len(h5_files)}개")
    print(f"  Ground truth 있는 파일: {files_with_gt}개")
    print(f"  Ground truth 없는 파일: {files_without_gt}개")
    print(f"  정상 파일 (normal_*): {normal_files}개")
    print(f"  이상 파일 (abnormal_*): {abnormal_files}개")
    
    print(f"\n🏷️ 라벨 통계:")
    print(f"  정상 프레임 (라벨=0): {total_normal_frames:,}개")
    print(f"  이상 프레임 (라벨=1): {total_abnormal_frames:,}개")
    print(f"  총 프레임: {total_normal_frames + total_abnormal_frames:,}개")
    
    if total_normal_frames + total_abnormal_frames > 0:
        normal_ratio = total_normal_frames / (total_normal_frames + total_abnormal_frames) * 100
        abnormal_ratio = total_abnormal_frames / (total_normal_frames + total_abnormal_frames) * 100
        print(f"  정상:이상 비율 = {normal_ratio:.1f}%:{abnormal_ratio:.1f}%")
    
    # 파일별 상세 분석
    if file_details:
        print(f"\n📈 파일별 라벨 분포:")
        
        # 정상 파일들의 라벨 분포
        normal_file_details = [f for f in file_details if not f['is_abnormal']]
        if normal_file_details:
            normal_0_frames = sum(f['normal_frames'] for f in normal_file_details)
            normal_1_frames = sum(f['abnormal_frames'] for f in normal_file_details)
            print(f"  정상 파일들 ({len(normal_file_details)}개):")
            print(f"    라벨=0 프레임: {normal_0_frames:,}개")
            print(f"    라벨=1 프레임: {normal_1_frames:,}개")
        
        # 이상 파일들의 라벨 분포
        abnormal_file_details = [f for f in file_details if f['is_abnormal']]
        if abnormal_file_details:
            abnormal_0_frames = sum(f['normal_frames'] for f in abnormal_file_details)
            abnormal_1_frames = sum(f['abnormal_frames'] for f in abnormal_file_details)
            print(f"  이상 파일들 ({len(abnormal_file_details)}개):")
            print(f"    라벨=0 프레임: {abnormal_0_frames:,}개")
            print(f"    라벨=1 프레임: {abnormal_1_frames:,}개")
    
    # 라벨 분포 히스토그램 (선택적)
    if label_distribution and len(set(label_distribution)) > 1:
        unique_labels, counts = np.unique(label_distribution, return_counts=True)
        print(f"\n📊 전체 라벨 분포:")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(label_distribution) * 100
            print(f"  라벨 {label}: {count:,}개 ({percentage:.1f}%)")
    
    return {
        'total_files': len(h5_files),
        'files_with_gt': files_with_gt,
        'normal_files': normal_files,
        'abnormal_files': abnormal_files,
        'normal_frames': total_normal_frames,
        'abnormal_frames': total_abnormal_frames,
        'file_details': file_details
    }


def visualize_keypoints(keypoints, frame_idx=0, title="Keypoints Visualization"):
    """키포인트 시각화
    
    Args:
        keypoints: 키포인트 데이터 (frames, keypoints, 3)
        frame_idx: 시각화할 프레임 인덱스
        title: 플롯 제목
    """
    
    if len(keypoints.shape) != 3 or keypoints.shape[2] != 3:
        print("Invalid keypoints shape. Expected (frames, keypoints, 3)")
        return
    
    if frame_idx >= keypoints.shape[0]:
        print(f"Frame index {frame_idx} out of range. Max: {keypoints.shape[0]-1}")
        return
    
    frame_keypoints = keypoints[frame_idx]
    
    # 유효한 키포인트만 필터링 (confidence > 0)
    valid_keypoints = frame_keypoints[frame_keypoints[:, 2] > 0]
    
    if len(valid_keypoints) == 0:
        print("No valid keypoints found in this frame")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 키포인트 시각화
    x_coords = valid_keypoints[:, 0]
    y_coords = valid_keypoints[:, 1]
    confidences = valid_keypoints[:, 2]
    
    # 스케터 플롯 - confidence에 따라 색상과 크기 조절
    scatter = plt.scatter(x_coords, y_coords, c=confidences, s=confidences*100, 
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # 키포인트 번호 표시
    for i, (x, y, conf) in enumerate(valid_keypoints):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='red', fontweight='bold')
    
    plt.colorbar(scatter, label='Confidence')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(f"{title} - Frame {frame_idx}")
    plt.grid(True, alpha=0.3)
    
    # Y축 뒤집기 (이미지 좌표계)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_keypoint_trajectory(keypoints, keypoint_idx=0, title="Keypoint Trajectory"):
    """특정 키포인트의 시간에 따른 궤적 시각화
    
    Args:
        keypoints: 키포인트 데이터 (frames, keypoints, 3)
        keypoint_idx: 추적할 키포인트 인덱스
        title: 플롯 제목
    """
    
    if keypoint_idx >= keypoints.shape[1]:
        print(f"Keypoint index {keypoint_idx} out of range. Max: {keypoints.shape[1]-1}")
        return
    
    # 특정 키포인트의 궤적 추출
    trajectory = keypoints[:, keypoint_idx, :]
    
    # 유효한 프레임만 필터링
    valid_frames = trajectory[:, 2] > 0
    valid_trajectory = trajectory[valid_frames]
    
    if len(valid_trajectory) == 0:
        print(f"No valid data found for keypoint {keypoint_idx}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # X 좌표 변화
    axes[0, 0].plot(valid_trajectory[:, 0], 'b.-', alpha=0.7)
    axes[0, 0].set_title(f'X coordinate over time - Keypoint {keypoint_idx}')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('X coordinate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Y 좌표 변화
    axes[0, 1].plot(valid_trajectory[:, 1], 'r.-', alpha=0.7)
    axes[0, 1].set_title(f'Y coordinate over time - Keypoint {keypoint_idx}')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Y coordinate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confidence 변화
    axes[1, 0].plot(valid_trajectory[:, 2], 'g.-', alpha=0.7)
    axes[1, 0].set_title(f'Confidence over time - Keypoint {keypoint_idx}')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Confidence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 2D 궤적
    axes[1, 1].plot(valid_trajectory[:, 0], valid_trajectory[:, 1], 'purple', alpha=0.7, linewidth=2)
    axes[1, 1].scatter(valid_trajectory[0, 0], valid_trajectory[0, 1], c='green', s=100, marker='o', label='Start')
    axes[1, 1].scatter(valid_trajectory[-1, 0], valid_trajectory[-1, 1], c='red', s=100, marker='x', label='End')
    axes[1, 1].set_title(f'2D Trajectory - Keypoint {keypoint_idx}')
    axes[1, 1].set_xlabel('X coordinate')
    axes[1, 1].set_ylabel('Y coordinate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].invert_yaxis()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze HDF5 keypoint data')
    parser.add_argument('--hdf5_dir', type=str, default='dataset_hdf5',
                      help='Directory containing HDF5 files')
    parser.add_argument('--file', type=str, help='Specific HDF5 file to analyze')
    parser.add_argument('--person_id', type=str, help='Specific person ID to analyze')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--stats_only', action='store_true', help='Only show statistics')
    parser.add_argument('--test_only', action='store_true', help='Analyze only test data ground truth')
    parser.add_argument('--gt_only', action='store_true', help='Analyze only ground truth labels')
    
    args = parser.parse_args()
    
    if args.file:
        # 특정 파일 분석
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        
        # 파일 구조 탐색
        explore_hdf5_structure(file_path)
        
        if not args.stats_only:
            # 데이터 로드
            data = load_keypoints_data(file_path, args.person_id)
            
            # 데이터 정보 출력
            print(f"\n=== Data Information ===")
            for person_id, person_data in data.items():
                if person_id == 'ground_truth':
                    continue
                    
                print(f"\nPerson: {person_id}")
                print(f"  Total frames: {person_data['total_frames']}")
                print(f"  Keypoints per frame: {person_data['num_keypoints']}")
                print(f"  Keypoints shape: {person_data['keypoints'].shape}")
                
                if 'scores' in person_data:
                    print(f"  Scores shape: {person_data['scores'].shape}")
            
            if data['ground_truth'] is not None:
                print(f"\nGround Truth: {data['ground_truth'].shape}")
            
            # 시각화
            if args.visualize and not args.stats_only:
                for person_id, person_data in data.items():
                    if person_id == 'ground_truth':
                        continue
                    
                    keypoints = person_data['keypoints']
                    if len(keypoints) > 0:
                        print(f"\nVisualizing data for {person_id}...")
                        
                        # 첫 번째 프레임 키포인트 시각화
                        visualize_keypoints(keypoints, frame_idx=0, 
                                          title=f"{person_id} - Keypoints")
                        
                        # 첫 번째 키포인트의 궤적 시각화
                        if keypoints.shape[1] > 0:
                            plot_keypoint_trajectory(keypoints, keypoint_idx=0,
                                                   title=f"{person_id} - Keypoint Trajectory")
                        
                        break  # 첫 번째 사람만 시각화
    
    else:
        # 전체 데이터셋 통계
        hdf5_dir = Path(args.hdf5_dir)
        
        if args.test_only or args.gt_only:
            # 테스트 데이터만 분석
            test_dir = hdf5_dir / "test"
            if test_dir.exists():
                print("=== Test Data Ground Truth Analysis ===")
                if args.gt_only:
                    analyze_ground_truth_labels(test_dir)
                else:
                    analyze_dataset_statistics(test_dir)
                    analyze_ground_truth_labels(test_dir)
            else:
                print(f"Test directory not found: {test_dir}")
        else:
            # 전체 데이터셋 분석
            # Train 데이터 분석
            train_dir = hdf5_dir / "train"
            if train_dir.exists():
                print("=== Training Data Analysis ===")
                analyze_dataset_statistics(train_dir)
                analyze_ground_truth_labels(train_dir)
            
            # Test 데이터 분석
            test_dir = hdf5_dir / "test"
            if test_dir.exists():
                print("\n=== Test Data Analysis ===")
                analyze_dataset_statistics(test_dir)
                analyze_ground_truth_labels(test_dir)


if __name__ == "__main__":
    main()