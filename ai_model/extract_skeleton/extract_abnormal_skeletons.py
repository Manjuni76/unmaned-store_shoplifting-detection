"""
mlp_train_data.json의 abnormal 데이터에서 skeleton 추출
"""
import json
import os
from pathlib import Path
from yolo_skeleton_extractor import YOLOSkeletonExtractor

def main():
    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    mlp_train_json = base_dir / 'data_split' / 'output' / 'mlp_train_data.json'
    output_dir = base_dir / 'data' / 'mlp_train_data_skeleton_data'
    
    print(f"[INFO] Reading: {mlp_train_json}")
    print(f"[INFO] Output dir: {output_dir}")
    
    # JSON 로드
    with open(mlp_train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Abnormal 데이터만 필터링
    abnormal_videos = data.get('abnormal', [])
    print(f"[INFO] Found {len(abnormal_videos)} abnormal videos")
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLO Skeleton Extractor 초기화
    extractor = YOLOSkeletonExtractor('yolov8n-pose.pt')
    
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, video_info in enumerate(abnormal_videos):
        filename = video_info['filename']
        video_path = video_info['full_path']
        
        # 출력 파일명 생성 (filename.mp4 -> filename_skeleton.json)
        output_filename = filename.replace('.mp4', '_skeleton.json')
        output_path = output_dir / output_filename
        
        print(f"\n[{idx+1}/{len(abnormal_videos)}] Processing: {filename}")
        
        # 이미 처리된 파일이 있으면 스킵
        if output_path.exists():
            print(f"  ✅ Already exists: {output_filename}")
            skipped += 1
            continue
        
        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            print(f"  ❌ Video not found: {video_path}")
            failed += 1
            continue
        
        try:
            # Skeleton 추출 (전체 프레임)
            skeleton_data = extractor.process_video(
                video_path=video_path,
                output_path=str(output_path),
                start_frame=0,
                end_frame=None
            )
            
            if skeleton_data:
                print(f"  ✅ Success: {output_filename}")
                successful += 1
            else:
                print(f"  ⚠️  Warning: No skeleton data extracted")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            failed += 1
            continue
    
    # 결과 요약
    print("\n" + "="*80)
    print("스켈레톤 추출 완료!")
    print("="*80)
    print(f"✅ Successful: {successful}")
    print(f"⏭️  Skipped (already exists): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
