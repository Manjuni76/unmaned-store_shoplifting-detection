import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random
import json
from collections import defaultdict
import math

def parse_xml_for_theft_frames(xml_path):
    """XML 파일에서 도난(theft) 시작과 끝 프레임을 추출합니다."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        theft_start = None
        theft_end = None
        
        for track in root.findall('.//track'):
            label = track.get('label')
            if label == 'theft_start':
                box = track.find('box')
                if box is not None:
                    frame_attr = box.get('frame')
                    if frame_attr is not None:
                        theft_start = int(frame_attr)
            elif label == 'theft_end':
                box = track.find('box')
                if box is not None:
                    frame_attr = box.get('frame')
                    if frame_attr is not None:
                        theft_end = int(frame_attr)
        
        return theft_start, theft_end
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return None, None

def count_video_frames(video_path):
    """비디오 파일의 프레임 수를 계산합니다."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return 0

def find_corresponding_video(xml_path, video_dir):
    """XML 파일명에 대응하는 비디오 파일을 찾습니다."""
    xml_name = Path(xml_path).stem
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    for ext in video_extensions:
        video_path = os.path.join(video_dir, xml_name + ext)
        if os.path.exists(video_path):
            return video_path
    
    # 재귀적으로 찾기
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                if xml_name in file or Path(file).stem in xml_name:
                    return os.path.join(root, file)
    
    return None

def extract_category_from_filename(filename):
    """파일명에서 카테고리를 추출합니다."""
    parts = filename.split('_')
    if len(parts) >= 6:
        category = parts[5]
        return category
    return "UNKNOWN"

def collect_normal_videos():
    """정상 비디오 데이터를 수집합니다."""
    normal_video_dir = "D:\\AI-HUB_shoping"
    normal_videos = []
    
    print("정상 비디오 수집 중...")
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    for root, dirs, files in os.walk(normal_video_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, file)
                frame_count = count_video_frames(video_path)
                if frame_count > 0:
                    category = extract_category_from_filename(file)
                    normal_videos.append({
                        'path': video_path,
                        'frames': frame_count,
                        'category': category,
                        'type': 'normal'
                    })
    
    return normal_videos

def collect_abnormal_videos():
    """이상 비디오 데이터를 수집합니다."""
    abnormal_video_dir = "D:\\AI-HUB_shoplifting"
    abnormal_videos = []
    
    print("이상 비디오 수집 중...")
    
    xml_files = []
    for root, dirs, files in os.walk(abnormal_video_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    for xml_path in xml_files:
        theft_start, theft_end = parse_xml_for_theft_frames(xml_path)
        
        if theft_start is None or theft_end is None:
            continue
        
        video_path = find_corresponding_video(xml_path, abnormal_video_dir)
        
        if video_path is None:
            continue
        
        total_video_frames = count_video_frames(video_path)
        
        if total_video_frames == 0:
            continue
        
        abnormal_frames = theft_end - theft_start + 1
        normal_frames_in_video = total_video_frames - abnormal_frames
        
        category = extract_category_from_filename(Path(video_path).name)
        
        abnormal_videos.append({
            'path': video_path,
            'xml_path': xml_path,
            'total_frames': total_video_frames,
            'theft_start': theft_start,
            'theft_end': theft_end,
            'normal_frames': normal_frames_in_video,
            'abnormal_frames': abnormal_frames,
            'category': category,
            'type': 'abnormal'
        })
    
    return abnormal_videos

def calculate_balanced_split_sizes():
    """균형잡힌 분할 크기를 계산합니다."""
    # 도난 데이터 제약으로 인한 MLP, Test 크기
    total_abnormal_frames = 48189
    
    # MLP:Test = 6:4 비율로 도난 데이터 분할
    mlp_abnormal = int(total_abnormal_frames * 0.6)  # 약 28,913
    test_abnormal = total_abnormal_frames - mlp_abnormal  # 약 19,276
    
    # 7:3 비율에 맞춰 정상 데이터 계산
    mlp_normal = int(mlp_abnormal * 7 / 3)  # 약 67,463
    test_normal = int(test_abnormal * 7 / 3)  # 약 44,977
    
    mlp_total = mlp_normal + mlp_abnormal
    test_total = test_normal + test_abnormal
    
    # 특징추출용은 MLP + Test 합계와 비슷하게
    feature_total = mlp_total + test_total  # 약 160,000 프레임
    
    print(f"균형잡힌 분할 계획:")
    print(f"1) 특징추출용: {feature_total:,} 프레임 (정상 데이터만, 카테고리 균등)")
    print(f"2) MLP 학습용: {mlp_total:,} 프레임 (정상:{mlp_normal:,}, 도난:{mlp_abnormal:,})")
    print(f"3) 테스트용: {test_total:,} 프레임 (정상:{test_normal:,}, 도난:{test_abnormal:,})")
    
    return {
        'feature_total': feature_total,
        'mlp_normal': mlp_normal,
        'mlp_abnormal': mlp_abnormal,
        'test_normal': test_normal,
        'test_abnormal': test_abnormal
    }

def split_data_balanced(normal_videos, abnormal_videos, split_sizes):
    """균형잡힌 데이터 분할을 수행합니다."""
    print("\n균형잡힌 데이터 분할 시작...")
    
    # 카테고리별로 그룹화
    normal_by_category = defaultdict(list)
    abnormal_by_category = defaultdict(list)
    
    for video in normal_videos:
        normal_by_category[video['category']].append(video)
    
    for video in abnormal_videos:
        abnormal_by_category[video['category']].append(video)
    
    print(f"정상 데이터 카테고리: {list(normal_by_category.keys())}")
    print(f"도난 데이터 카테고리: {list(abnormal_by_category.keys())}")
    
    # 모든 데이터를 한 번에 섞어서 중복 방지
    all_normal_videos = []
    for category, videos in normal_by_category.items():
        all_normal_videos.extend(videos)
    
    random.shuffle(all_normal_videos)
    
    # 2), 3)에서 사용할 도난 데이터 먼저 분할
    all_abnormal_segments = []
    for category, videos in abnormal_by_category.items():
        for video in videos:
            # 정상 구간들
            if video['theft_start'] > 0:
                all_abnormal_segments.append({
                    'video': video,
                    'start_frame': 0,
                    'end_frame': video['theft_start'] - 1,
                    'frames': video['theft_start'],
                    'label': 'normal'
                })
            
            if video['theft_end'] < video['total_frames'] - 1:
                all_abnormal_segments.append({
                    'video': video,
                    'start_frame': video['theft_end'] + 1,
                    'end_frame': video['total_frames'] - 1,
                    'frames': video['total_frames'] - video['theft_end'] - 1,
                    'label': 'normal'
                })
            
            # 도난 구간
            all_abnormal_segments.append({
                'video': video,
                'start_frame': video['theft_start'],
                'end_frame': video['theft_end'],
                'frames': video['abnormal_frames'],
                'label': 'abnormal'
            })
    
    random.shuffle(all_abnormal_segments)
    
    # MLP용 데이터 선택
    mlp_data = []
    mlp_abnormal_needed = split_sizes['mlp_abnormal']
    mlp_normal_needed = split_sizes['mlp_normal']
    
    current_mlp_abnormal = 0
    current_mlp_normal = 0
    used_segments = []
    
    for segment in all_abnormal_segments:
        if segment['label'] == 'abnormal' and current_mlp_abnormal < mlp_abnormal_needed:
            needed = min(mlp_abnormal_needed - current_mlp_abnormal, segment['frames'])
            mlp_data.append({
                'video': segment['video'],
                'start_frame': segment['start_frame'],
                'end_frame': segment['start_frame'] + needed - 1,
                'frames_to_use': needed,
                'label': 'abnormal'
            })
            current_mlp_abnormal += needed
            used_segments.append(segment)
            
        elif segment['label'] == 'normal' and current_mlp_normal < mlp_normal_needed:
            needed = min(mlp_normal_needed - current_mlp_normal, segment['frames'])
            mlp_data.append({
                'video': segment['video'],
                'start_frame': segment['start_frame'],
                'end_frame': segment['start_frame'] + needed - 1,
                'frames_to_use': needed,
                'label': 'normal'
            })
            current_mlp_normal += needed
            used_segments.append(segment)
        
        if current_mlp_abnormal >= mlp_abnormal_needed and current_mlp_normal >= mlp_normal_needed:
            break
    
    # 테스트용 데이터 선택 (남은 것들로)
    test_data = []
    test_abnormal_needed = split_sizes['test_abnormal']
    test_normal_needed = split_sizes['test_normal']
    
    current_test_abnormal = 0
    current_test_normal = 0
    
    for segment in all_abnormal_segments:
        if segment in used_segments:
            continue
            
        if segment['label'] == 'abnormal' and current_test_abnormal < test_abnormal_needed:
            needed = min(test_abnormal_needed - current_test_abnormal, segment['frames'])
            test_data.append({
                'video': segment['video'],
                'start_frame': segment['start_frame'],
                'end_frame': segment['start_frame'] + needed - 1,
                'frames_to_use': needed,
                'label': 'abnormal'
            })
            current_test_abnormal += needed
            used_segments.append(segment)
            
        elif segment['label'] == 'normal' and current_test_normal < test_normal_needed:
            needed = min(test_normal_needed - current_test_normal, segment['frames'])
            test_data.append({
                'video': segment['video'],
                'start_frame': segment['start_frame'],
                'end_frame': segment['start_frame'] + needed - 1,
                'frames_to_use': needed,
                'label': 'normal'
            })
            current_test_normal += needed
            used_segments.append(segment)
        
        if current_test_abnormal >= test_abnormal_needed and current_test_normal >= test_normal_needed:
            break
    
    # 사용된 정상 비디오들 제거 (MLP, Test에서 사용된 정상 구간)
    used_normal_videos = set()
    for segment in used_segments:
        if segment['label'] == 'normal':
            used_normal_videos.add(segment['video']['path'])
    
    # MLP, Test 부족분을 일반 정상 비디오로 채우기
    remaining_normal = [v for v in all_normal_videos if v['path'] not in used_normal_videos]
    random.shuffle(remaining_normal)
    
    # MLP 정상 데이터 부족분 채우기
    if current_mlp_normal < mlp_normal_needed:
        needed = mlp_normal_needed - current_mlp_normal
        for video in remaining_normal:
            if needed <= 0:
                break
            frames_to_take = min(needed, video['frames'])
            mlp_data.append({
                'video': video,
                'start_frame': 0,
                'end_frame': frames_to_take - 1,
                'frames_to_use': frames_to_take,
                'label': 'normal'
            })
            needed -= frames_to_take
            used_normal_videos.add(video['path'])
    
    # 테스트 정상 데이터 부족분 채우기
    remaining_normal = [v for v in remaining_normal if v['path'] not in used_normal_videos]
    if current_test_normal < test_normal_needed:
        needed = test_normal_needed - current_test_normal
        for video in remaining_normal:
            if needed <= 0:
                break
            frames_to_take = min(needed, video['frames'])
            test_data.append({
                'video': video,
                'start_frame': 0,
                'end_frame': frames_to_take - 1,
                'frames_to_use': frames_to_take,
                'label': 'normal'
            })
            needed -= frames_to_take
            used_normal_videos.add(video['path'])
    
    # 1) 특징추출용 데이터 (남은 정상 데이터로, 카테고리 균등)
    remaining_normal = [v for v in all_normal_videos if v['path'] not in used_normal_videos]
    
    # 카테고리별로 재그룹화
    remaining_by_category = defaultdict(list)
    for video in remaining_normal:
        remaining_by_category[video['category']].append(video)
    
    feature_data = []
    feature_frames_needed = split_sizes['feature_total']
    categories = list(remaining_by_category.keys())
    frames_per_category = feature_frames_needed // len(categories)
    
    print(f"\n특징추출용 데이터 분할 (카테고리당 {frames_per_category:,} 프레임)...")
    
    for category in categories:
        current_frames = 0
        random.shuffle(remaining_by_category[category])
        
        for video in remaining_by_category[category]:
            if current_frames >= frames_per_category:
                break
            
            needed_frames = min(frames_per_category - current_frames, video['frames'])
            
            feature_data.append({
                'video': video,
                'frames_to_use': needed_frames,
                'start_frame': 0,
                'end_frame': needed_frames - 1,
                'label': 'normal'
            })
            
            current_frames += needed_frames
            
            if current_frames >= frames_per_category:
                break
    
    return feature_data, mlp_data, test_data

def save_data_splits(feature_data, mlp_data, test_data, output_dir):
    """분할된 데이터를 저장합니다."""
    print("\n데이터 분할 결과 저장 중...")
    
    # 기존 디렉토리 제거 후 재생성
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    feature_dir = os.path.join(output_dir, "train_feature")
    mlp_dir = os.path.join(output_dir, "train_mlp")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(mlp_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    def save_split_info(data, split_dir, split_name):
        info_file = os.path.join(split_dir, f"{split_name}_info.json")
        
        split_info = {
            'total_videos': len(set(item['video']['path'] for item in data)),
            'total_frames': sum(item['frames_to_use'] for item in data),
            'normal_frames': sum(item['frames_to_use'] for item in data if item['label'] == 'normal'),
            'abnormal_frames': sum(item['frames_to_use'] for item in data if item['label'] == 'abnormal'),
            'data': []
        }
        
        # 카테고리별 통계
        category_stats = {}
        for item in data:
            category = item['video'].get('category', 'UNKNOWN')
            if category not in category_stats:
                category_stats[category] = {'normal': 0, 'abnormal': 0, 'videos': set()}
            
            category_stats[category][item['label']] += item['frames_to_use']
            category_stats[category]['videos'].add(item['video']['path'])
        
        for item in data:
            split_info['data'].append({
                'video_path': item['video']['path'],
                'start_frame': item['start_frame'],
                'end_frame': item['end_frame'],
                'frames_to_use': item['frames_to_use'],
                'label': item['label'],
                'category': item['video'].get('category', 'UNKNOWN')
            })
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n{split_name} 저장 완료:")
        print(f"  - 총 비디오: {split_info['total_videos']}개")
        print(f"  - 총 프레임: {split_info['total_frames']:,}개")
        print(f"  - 정상 프레임: {split_info['normal_frames']:,}개")
        print(f"  - 도난 프레임: {split_info['abnormal_frames']:,}개")
        if split_info['abnormal_frames'] > 0:
            ratio = split_info['normal_frames'] / split_info['abnormal_frames']
            print(f"  - 정상:도난 비율: {ratio:.1f}:1")
        
        print(f"  - 카테고리별 분포:")
        for category, stats in sorted(category_stats.items()):
            total_cat_frames = stats['normal'] + stats['abnormal']
            print(f"    {category}: {len(stats['videos'])}개 비디오, "
                  f"{total_cat_frames:,}프레임 (정상:{stats['normal']:,}, 도난:{stats['abnormal']:,})")
    
    save_split_info(feature_data, feature_dir, "feature_extraction")
    save_split_info(mlp_data, mlp_dir, "mlp_training")
    save_split_info(test_data, test_dir, "test")

def main():
    print("="*60)
    print("균형잡힌 도난 탐지 데이터 분할")
    print("="*60)
    
    # 1. 데이터 수집
    normal_videos = collect_normal_videos()
    abnormal_videos = collect_abnormal_videos()
    
    print(f"\n수집된 데이터:")
    print(f"정상 비디오: {len(normal_videos)}개")
    print(f"이상 비디오: {len(abnormal_videos)}개")
    
    # 2. 균형잡힌 분할 크기 계산
    split_sizes = calculate_balanced_split_sizes()
    
    # 3. 데이터 분할
    feature_data, mlp_data, test_data = split_data_balanced(
        normal_videos, abnormal_videos, split_sizes)
    
    # 4. 결과 저장
    output_dir = "data_splits_balanced"
    save_data_splits(feature_data, mlp_data, test_data, output_dir)
    
    print("\n" + "="*60)
    print("균형잡힌 데이터 분할 완료!")
    print("="*60)

if __name__ == "__main__":
    main()