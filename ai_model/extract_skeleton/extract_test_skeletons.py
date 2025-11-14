"""
Test 데이터의 abnormal skeleton 추출 스크립트
"""

import json
import os
from yolo_skeleton_extractor import YOLOSkeletonExtractor

def main():
    # 경로 설정
    base_dir = r"C:\Users\User\Desktop\Sejong\Under_Graduate\3_2\파이썬기반딥러닝\unmaned_store_shoplifting_detection"
    test_json = os.path.join(base_dir, "data_split", "output", "test_data.json")
    output_dir = os.path.join(base_dir, "data", "test_data_skeleton_data")
    
    print(f"[INFO] Reading: {test_json}")
    
    # test_data.json 읽기
    with open(test_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # abnormal 비디오 가져오기
    abnormal_videos = data.get('abnormal', [])
    
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Found {len(abnormal_videos)} abnormal videos in test data")
    
    # 이미 존재하는 파일 확인
    existing_files = set()
    if os.path.exists(output_dir):
        existing_files = {f for f in os.listdir(output_dir) if f.endswith('_skeleton.json')}
    
    print(f"[INFO] Already have {len(existing_files)} skeleton files")
    
    # YOLO 추출기 초기화
    model_path = os.path.join(base_dir, "ai_model", "extract_skeleton", "yolov8n-pose.pt")
    extractor = YOLOSkeletonExtractor(model_path)
    
    # 각 abnormal 비디오에 대해 skeleton 추출
    success_count = 0
    skip_count = 0
    
    for idx, video_info in enumerate(abnormal_videos, 1):
        filename = video_info['filename']
        video_path = video_info['full_path']
        skeleton_filename = filename.replace('.mp4', '_skeleton.json')
        output_path = os.path.join(output_dir, skeleton_filename)
        
        print(f"\n[{idx}/{len(abnormal_videos)}] Processing: {filename}")
        
        # 이미 존재하면 스킵
        if skeleton_filename in existing_files:
            print(f"  ✅ Already exists: {skeleton_filename}")
            skip_count += 1
            continue
        
        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            print(f"  ❌ Video not found: {video_path}")
            continue
        
        try:
            # Skeleton 추출
            skeleton_data = extractor.process_video(
                video_path=video_path,
                start_frame=0,
                end_frame=None
            )
            
            # 저장
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Success: {skeleton_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    print(f"\n" + "="*50)
    print(f"[COMPLETE] Test abnormal skeleton extraction")
    print(f"  - Total: {len(abnormal_videos)}")
    print(f"  - Already existed: {skip_count}")
    print(f"  - Newly extracted: {success_count}")
    print(f"  - Total available: {skip_count + success_count}/{len(abnormal_videos)}")

if __name__ == "__main__":
    main()
