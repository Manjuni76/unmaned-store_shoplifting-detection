import cv2
import torch
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import argparse

class YOLOSkeletonExtractor:
    def __init__(self, model_path='yolov8n-pose.pt', conf_threshold=0.3, kpt_threshold=0.1):
        """
        YOLO Pose 기반 스켈레톤 추출기
        
        Args:
            model_path: YOLO pose 모델 경로 (yolov8n-pose.pt, yolov8s-pose.pt 등)
            conf_threshold: 사람 탐지 신뢰도 임계값
            kpt_threshold: 키포인트 신뢰도 임계값
        """
        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self.device = torch.device('cuda')  # 강제로 CUDA 사용
        print(f"Using device: {self.device}")
        
        # YOLO Pose 모델 로드
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)  # 모델을 GPU로 이동
            print(f"YOLO Pose model loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Downloading yolov8n-pose.pt...")
            self.model = YOLO('yolov8n-pose.pt')
        
        # COCO17 키포인트 순서 정의
        self.coco17_keypoints = [
            'nose',           # 0
            'left_eye',       # 1
            'right_eye',      # 2
            'left_ear',       # 3
            'right_ear',      # 4
            'left_shoulder',  # 5
            'right_shoulder', # 6
            'left_elbow',     # 7
            'right_elbow',    # 8
            'left_wrist',     # 9
            'right_wrist',    # 10
            'left_hip',       # 11
            'right_hip',      # 12
            'left_knee',      # 13
            'right_knee',     # 14
            'left_ankle',     # 15
            'right_ankle'     # 16
        ]
        
        self.confidence_threshold = 0.5  # 사람 탐지 신뢰도 임계값
        self.keypoint_threshold = 0.3    # 키포인트 신뢰도 임계값
    
    def extract_poses_from_frame(self, frame):
        """
        단일 프레임에서 모든 사람의 포즈 추출
        
        Args:
            frame: 입력 이미지 프레임
            
        Returns:
            list: 각 사람의 포즈 데이터 [{"keypoints": [[x,y,conf], ...], "bbox": [x1,y1,x2,y2], "conf": float}, ...]
        """
        # YOLO 추론 실행
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        poses = []
        
        if len(results) > 0 and results[0].keypoints is not None:
            # 키포인트 데이터 추출
            keypoints = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2) - N명의 사람, 17개 키포인트, x,y 좌표
            confidences = results[0].keypoints.conf.cpu().numpy()  # (N, 17) - 각 키포인트의 신뢰도
            
            # 바운딩 박스 정보
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4) - x1,y1,x2,y2
                box_confidences = results[0].boxes.conf.cpu().numpy()  # (N,) - 박스 신뢰도
            else:
                boxes = None
                box_confidences = None
            
            # 각 사람별로 처리
            for i in range(len(keypoints)):
                person_keypoints = []
                
                # 17개 키포인트 처리
                for j in range(17):
                    if j < len(keypoints[i]):
                        x, y = keypoints[i][j]
                        conf = confidences[i][j] if j < len(confidences[i]) else 0.0
                        
                        # 신뢰도가 낮거나 좌표가 유효하지 않으면 0으로 설정
                        if conf < self.kpt_threshold or x <= 0 or y <= 0:
                            person_keypoints.append([0.0, 0.0, 0.0])
                        else:
                            person_keypoints.append([float(x), float(y), float(conf)])
                    else:
                        person_keypoints.append([0.0, 0.0, 0.0])
                
                # 바운딩 박스 정보 추가
                bbox = [0, 0, 0, 0]
                bbox_conf = 0.0
                
                if boxes is not None and i < len(boxes):
                    bbox = [float(x) for x in boxes[i]]
                    bbox_conf = float(box_confidences[i]) if box_confidences is not None else 0.0
                
                # 유효한 키포인트가 충분한 경우만 추가 (최소 5개 이상)
                valid_keypoints = sum(1 for kp in person_keypoints if kp[2] > 0)
                if valid_keypoints >= 5:  # 최소 5개 키포인트가 유효해야 함
                    poses.append({
                        "keypoints": person_keypoints,
                        "bbox": bbox,
                        "bbox_confidence": bbox_conf
                    })
        
        return poses
    
    def process_video(self, video_path, output_path=None, start_frame=0, end_frame=None):
        """
        비디오에서 스켈레톤 데이터 추출
        
        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 JSON 파일 경로
            start_frame: 시작 프레임 (기본값: 0)
            end_frame: 끝 프레임 (기본값: None - 끝까지)
            
        Returns:
            dict: 추출된 스켈레톤 데이터
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if end_frame is None:
            end_frame = total_frames - 1
        
        print(f"Processing video: {Path(video_path).name}")
        print(f"Frames: {start_frame}-{end_frame} (Total: {total_frames}, FPS: {fps:.2f})")
        
        # 시작 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        skeleton_data = {}
        frame_idx = start_frame
        
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 포즈 추출
            poses = self.extract_poses_from_frame(frame)
            
            # 프레임별 데이터 저장
            for person_idx, pose_data in enumerate(poses):
                person_id = f"person_{person_idx + 1}"
                
                if person_id not in skeleton_data:
                    skeleton_data[person_id] = {}
                
                # 프레임 인덱스는 0부터 시작하도록 조정
                relative_frame = frame_idx - start_frame
                skeleton_data[person_id][str(relative_frame)] = {
                    "keypoints": pose_data["keypoints"]
                }
            
            frame_idx += 1
            
            # 진행 상황 출력
            if (frame_idx - start_frame) % 50 == 0:
                processed = frame_idx - start_frame + 1
                total_to_process = end_frame - start_frame + 1
                print(f"Processed {processed}/{total_to_process} frames ({processed/total_to_process*100:.1f}%)")
        
        cap.release()
        
        print(f"Extraction completed. Found {len(skeleton_data)} persons")
        
        # 결과 저장
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:  # 디렉토리가 있는 경우에만 생성
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        
        return skeleton_data
    
    def process_video_batch(self, video_info_json, output_dir):
        """
        분할된 비디오 리스트에서 스켈레톤 데이터 추출
        
        Args:
            video_info_json: data_splits_balanced의 JSON 파일 경로
            output_dir: 출력 디렉토리
        """
        # JSON 파일 로드
        with open(video_info_json, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing {len(video_data['data'])} video segments...")
        
        processed_videos = set()
        successful_extractions = 0
        
        for idx, video_info in enumerate(video_data['data']):
            video_path = video_info['video_path']
            start_frame = video_info['start_frame']
            end_frame = video_info['end_frame']
            label = video_info['label']
            category = video_info['category']
            
            # 중복 처리 방지
            video_key = f"{Path(video_path).stem}_{start_frame}_{end_frame}"
            if video_key in processed_videos:
                continue
            processed_videos.add(video_key)
            
            print(f"\n[{idx+1}/{len(video_data['data'])}] Processing: {Path(video_path).name}")
            print(f"Frames: {start_frame}-{end_frame}, Label: {label}, Category: {category}")
            
            try:
                # 비디오 파일이 존재하는지 확인
                if not os.path.exists(video_path):
                    print(f"⚠️  Warning: Video file not found: {video_path}")
                    continue
                
                # 출력 파일명 생성
                output_filename = f"{Path(video_path).stem}_f{start_frame}_{end_frame}_{label}.json"
                output_path = os.path.join(output_dir, output_filename)
                
                # 이미 처리된 파일이 있으면 스킵
                if os.path.exists(output_path):
                    print(f"✅ Already processed: {output_filename}")
                    successful_extractions += 1
                    continue
                
                # 스켈레톤 데이터 추출
                skeleton_data = self.process_video(
                    video_path, 
                    output_path, 
                    start_frame, 
                    end_frame
                )
                
                # 추출된 데이터 검증
                if skeleton_data:
                    print(f"✅ Completed: {output_filename}")
                    successful_extractions += 1
                else:
                    print(f"⚠️  No skeleton data extracted: {output_filename}")
                
            except Exception as e:
                print(f"❌ Error processing {Path(video_path).name}: {str(e)}")
                continue
        
        print(f"\n🎉 Batch processing completed!")
        print(f"✅ Successful extractions: {successful_extractions}")
        print(f"📁 Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract skeleton data using YOLO Pose')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input video file or JSON info file')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output directory or file')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single', 
                       help='Processing mode: single video or batch from JSON')
    parser.add_argument('--model', default='yolov8n-pose.pt', 
                       help='YOLO pose model (yolov8n-pose.pt, yolov8s-pose.pt, etc.)')
    parser.add_argument('--start_frame', type=int, default=0, 
                       help='Start frame for single video mode')
    parser.add_argument('--end_frame', type=int, default=None, 
                       help='End frame for single video mode')
    
    args = parser.parse_args()
    
    # 스켈레톤 추출기 초기화
    extractor = YOLOSkeletonExtractor(args.model)
    
    if args.mode == 'single':
        # 단일 비디오 처리
        skeleton_data = extractor.process_video(
            args.input, 
            args.output, 
            args.start_frame, 
            args.end_frame
        )
        
    elif args.mode == 'batch':
        # 배치 처리 (JSON 파일 기반)
        extractor.process_video_batch(args.input, args.output)

if __name__ == "__main__":
    main()