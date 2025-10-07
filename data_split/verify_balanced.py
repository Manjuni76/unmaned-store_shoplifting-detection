import json
import os

def verify_balanced_split():
    """균형잡힌 분할 결과를 검증합니다."""
    
    print("="*60)
    print("균형잡힌 데이터 분할 결과 검증")
    print("="*60)
    
    # 각 분할의 정보 파일 읽기
    splits = {
        'feature_extraction': 'data_splits_balanced/train_feature/feature_extraction_info.json',
        'mlp_training': 'data_splits_balanced/train_mlp/mlp_training_info.json',
        'test': 'data_splits_balanced/test/test_info.json'
    }
    
    split_stats = {}
    
    for split_name, file_path in splits.items():
        print(f"\n📊 {split_name.upper()} 분석:")
        print("-" * 40)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_frames = data['total_frames']
        normal_frames = data['normal_frames']
        abnormal_frames = data['abnormal_frames']
        total_videos = data['total_videos']
        
        split_stats[split_name] = {
            'total_frames': total_frames,
            'normal_frames': normal_frames,
            'abnormal_frames': abnormal_frames,
            'total_videos': total_videos
        }
        
        print(f"총 비디오 수: {total_videos}개")
        print(f"총 프레임 수: {total_frames:,}개")
        print(f"정상 프레임: {normal_frames:,}개 ({normal_frames/total_frames*100:.1f}%)")
        print(f"도난 프레임: {abnormal_frames:,}개 ({abnormal_frames/total_frames*100:.1f}%)")
        
        if abnormal_frames > 0:
            ratio = normal_frames / abnormal_frames
            print(f"정상:도난 비율: {ratio:.1f}:1")
    
    # 비교 분석
    print(f"\n" + "="*60)
    print("분할 비율 분석")
    print("="*60)
    
    feature_frames = split_stats['feature_extraction']['total_frames']
    mlp_frames = split_stats['mlp_training']['total_frames']
    test_frames = split_stats['test']['total_frames']
    
    total_frames = feature_frames + mlp_frames + test_frames
    
    feature_ratio = feature_frames / total_frames
    mlp_ratio = mlp_frames / total_frames
    test_ratio = test_frames / total_frames
    
    print(f"특징추출 : MLP : 테스트 비율")
    print(f"{feature_ratio:.3f} : {mlp_ratio:.3f} : {test_ratio:.3f}")
    print(f"≈ {feature_ratio*10:.1f} : {mlp_ratio*10:.1f} : {test_ratio*10:.1f}")
    
    print(f"\n프레임 수 비교:")
    print(f"특징추출용: {feature_frames:,}개")
    print(f"MLP 학습용: {mlp_frames:,}개")
    print(f"테스트용: {test_frames:,}개")
    print(f"총합: {total_frames:,}개")
    
    # MLP + Test vs Feature 비교
    mlp_test_total = mlp_frames + test_frames
    print(f"\nMLP + Test 합계: {mlp_test_total:,}개")
    print(f"특징추출용과 차이: {abs(feature_frames - mlp_test_total):,}개")
    print(f"비율: {feature_frames / mlp_test_total:.3f}")
    
    # 카테고리 균형 확인
    print(f"\n" + "="*60)
    print("카테고리 균형 확인 (특징추출용)")
    print("="*60)
    
    with open(splits['feature_extraction'], 'r', encoding='utf-8') as f:
        feature_data = json.load(f)
    
    category_frames = {}
    for item in feature_data['data']:
        category = item['category']
        frames = item['frames_to_use']
        
        if category not in category_frames:
            category_frames[category] = 0
        category_frames[category] += frames
    
    print("카테고리별 프레임 수:")
    for category, frames in sorted(category_frames.items()):
        print(f"  {category}: {frames:,}개")
    
    # 균등성 체크
    frame_counts = list(category_frames.values())
    min_frames = min(frame_counts)
    max_frames = max(frame_counts)
    
    print(f"\n균등성 분석:")
    print(f"최소: {min_frames:,}개")
    print(f"최대: {max_frames:,}개")
    print(f"차이: {max_frames - min_frames:,}개")
    print(f"균등도: {min_frames/max_frames*100:.1f}%")

if __name__ == "__main__":
    verify_balanced_split()