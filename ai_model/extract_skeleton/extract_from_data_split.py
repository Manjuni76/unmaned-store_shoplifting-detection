"""
data_split 출력 기반 고성능 스켈레톤 추출 스크립트
YOLOv11-pose 또는 YOLOv8x-pose를 사용하여 정확도 향상
멀티프로세싱으로 여러 비디오 동시 처리
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import argparse
import multiprocessing as mp
from functools import partial


class ImprovedSkeletonExtractor:
    def __init__(self, model_name='yolov8x-pose.pt', conf_threshold=0.5, kpt_threshold=0.3, device='cuda'):
        """
        개선된 스켈레톤 추출기
        
        Args:
            model_name: 모델 선택
                - 'yolov8x-pose.pt': YOLOv8 extra-large (가장 정확함)
                - 'yolov8l-pose.pt': YOLOv8 large
                - 'yolov8m-pose.pt': YOLOv8 medium
                - 'yolo11n-pose.pt': YOLOv11 nano (최신)
                - 'yolo11s-pose.pt': YOLOv11 small
                - 'yolo11m-pose.pt': YOLOv11 medium
                - 'yolo11l-pose.pt': YOLOv11 large
                - 'yolo11x-pose.pt': YOLOv11 extra-large (최고 성능)
            conf_threshold: 사람 탐지 신뢰도 임계값 (기본 0.5)
            kpt_threshold: 키포인트 신뢰도 임계값 (기본 0.3)
            device: 'cuda' or 'cpu'
        """
        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self.device = device
        
        print(f"[INFO] Loading {model_name}...")
        print(f"[INFO] Device: {device}")
        
        # YOLO Pose 모델 로드
        self.model = YOLO(model_name)
        
        # 디바이스 설정 - YOLO의 경우 to() 대신 predict/train에서 device 지정
        if device == 'cuda':
            try:
                # GPU가 사용 가능한지 확인
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
                    self.model.to('cuda')
                    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print(f"[WARNING] CUDA not available, using CPU instead")
                    self.device = 'cpu'
            except Exception as e:
                print(f"[WARNING] Failed to use GPU: {e}, using CPU instead")
                self.device = 'cpu'
        
        print(f"[SUCCESS] Model loaded: {model_name}")
        
        # COCO17 키포인트
        self.coco17_keypoints = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def extract_poses_from_frame(self, frame):
        """
        프레임에서 모든 사람의 포즈 추출
        
        Returns:
            list: [{"keypoints": [[x,y,conf], ...], "bbox": [x1,y1,x2,y2], "box_conf": float}]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        poses = []
        
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
            confidences = results[0].keypoints.conf.cpu().numpy()  # (N, 17)
            
            boxes = None
            box_confidences = None
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
                box_confidences = results[0].boxes.conf.cpu().numpy()  # (N,)
            
            for i in range(len(keypoints)):
                person_keypoints = []
                
                for j in range(17):
                    if j < len(keypoints[i]):
                        x, y = keypoints[i][j]
                        conf = confidences[i][j] if j < len(confidences[i]) else 0.0
                        
                        # 신뢰도 낮으면 0으로
                        if conf < self.kpt_threshold or x <= 0 or y <= 0:
                            person_keypoints.append([0.0, 0.0, 0.0])
                        else:
                            person_keypoints.append([float(x), float(y), float(conf)])
                    else:
                        person_keypoints.append([0.0, 0.0, 0.0])
                
                person_data = {
                    "keypoints": person_keypoints
                }
                
                if boxes is not None and i < len(boxes):
                    person_data["bbox"] = boxes[i].tolist()
                    person_data["box_conf"] = float(box_confidences[i])
                
                poses.append(person_data)
        
        return poses
    
    def extract_video_skeleton(self, video_path, output_json_path, max_persons=5):
        """
        비디오에서 스켈레톤 추출 및 저장 (dataset.py 형식에 맞춤)
        
        Args:
            video_path: 입력 비디오 경로
            output_json_path: 출력 JSON 경로
            max_persons: 프레임당 최대 인원 (confidence 높은 순, 기본적으로 person_1만 사용)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # dataset.py가 기대하는 형식: {"person_1": {"0": {"keypoints": [...]}}}
        skeleton_data = {
            "person_1": {}
        }
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(str(video_path))}", leave=False)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 포즈 추출
            poses = self.extract_poses_from_frame(frame)
            
            # confidence 높은 순으로 정렬
            if len(poses) > 0 and 'box_conf' in poses[0]:
                poses = sorted(poses, key=lambda x: x.get('box_conf', 0), reverse=True)
            
            # 가장 신뢰도 높은 사람만 사용 (person_1)
            if len(poses) > 0:
                skeleton_data["person_1"][str(frame_idx)] = {
                    "keypoints": poses[0]["keypoints"]
                }
            else:
                # 사람이 없으면 빈 키포인트
                skeleton_data["person_1"][str(frame_idx)] = {
                    "keypoints": [[0.0, 0.0, 0.0]] * 17
                }
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # JSON 저장
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
        
        return True


def load_json_split(json_path):
    """data_split 출력 JSON 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    videos = []
    for item in data.get('normal', []):
        item['category'] = 'normal'
        videos.append(item)
    
    for item in data.get('abnormal', []):
        item['category'] = 'abnormal'
        videos.append(item)
    
    return videos


def process_single_video(args_tuple):
    """
    단일 비디오 처리 함수 (멀티프로세싱용)
    
    Args:
        args_tuple: (video_info, output_skeleton_dir, model_name, conf, kpt_conf, device, max_persons)
    
    Returns:
        tuple: (status, filename) - status는 'success', 'skip', 'fail' 중 하나
    """
    video_info, output_skeleton_dir, model_name, conf, kpt_conf, device, max_persons, normal_video_root, abnormal_video_root = args_tuple
    
    filename = video_info['filename']
    category = video_info['category']
    
    # full_path가 있으면 사용, 없으면 기존 방식
    if 'full_path' in video_info:
        video_path = Path(video_info['full_path'])
    else:
        # 비디오 경로 (Path 객체 사용)
        if category == 'normal':
            video_path = normal_video_root / filename
        else:
            video_path = abnormal_video_root / filename
    
    # 출력 JSON 경로 (원본 파일명 유지, 확장자만 .json으로)
    skeleton_filename = filename.replace('.mp4', '.json')
    output_json_path = output_skeleton_dir / skeleton_filename
    
    # 이미 존재하면 스킵
    if output_json_path.exists():
        return ('skip', filename)
    
    # 비디오 존재 확인
    if not video_path.exists():
        return ('fail', filename)
    
    # 프로세스별로 모델 로드 (각 프로세스가 자신의 모델 인스턴스를 가짐)
    try:
        extractor = ImprovedSkeletonExtractor(
            model_name=model_name,
            conf_threshold=conf,
            kpt_threshold=kpt_conf,
            device=device
        )
        
        # 스켈레톤 추출
        success = extractor.extract_video_skeleton(
            video_path=video_path,
            output_json_path=str(output_json_path),
            max_persons=max_persons
        )
        
        if success:
            return ('success', filename)
        else:
            return ('fail', filename)
    except Exception as e:
        print(f"[ERROR] Exception processing {filename}: {e}")
        return ('fail', filename)


def main():
    parser = argparse.ArgumentParser(description='Extract skeleton using improved YOLO pose model with multiprocessing')
    parser.add_argument('--model', type=str, default='yolov8x-pose.pt',
                        choices=[
                            'yolov8x-pose.pt', 'yolov8l-pose.pt', 'yolov8m-pose.pt',
                            'yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt',
                            'yolo11l-pose.pt', 'yolo11x-pose.pt'
                        ],
                        help='YOLO pose model to use (yolov8x-pose.pt or yolo11x-pose.pt recommended)')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--kpt_conf', type=float, default=0.3, help='Keypoint confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--max_persons', type=int, default=5, help='Max persons per frame')
    parser.add_argument('--data_type', type=str, default='all', 
                        choices=['train', 'mlp_train', 'test', 'all'],
                        help='Which dataset to extract')
    parser.add_argument('--num_workers', type=int, default=-1, 
                        help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    data_split_dir = base_dir / "data_split" / "output"
    
    # 비디오 루트 경로 (Path 객체로 변환)
    normal_video_root = Path("D:/AI-HUB_shoping/shoping_data/Training/raw_data")
    abnormal_video_root = Path("D:/AI-HUB_shoplifting/shoplift_data/Training/raw_data/Shoplift")
    
    # 처리할 데이터 결정
    datasets_to_process = []
    if args.data_type in ['train', 'all']:
        datasets_to_process.append(('train', 'train_data.json', 'train_feature'))
    if args.data_type in ['mlp_train', 'all']:
        datasets_to_process.append(('mlp_train', 'mlp_train_data.json', 'train_mlp'))
    if args.data_type in ['test', 'all']:
        datasets_to_process.append(('test', 'test_data.json', 'test'))
    
    print(f"\n[INFO] Using {args.num_workers} parallel workers")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Model: {args.model}")
    
    for dataset_name, json_filename, skeleton_dir_name in datasets_to_process:
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        json_path = data_split_dir / json_filename
        if not json_path.exists():
            print(f"[WARNING] JSON not found: {json_path}")
            continue
        
        # 출력 디렉토리
        output_skeleton_dir = base_dir / "skeleton_extracted" / skeleton_dir_name / "skeleton_data"
        output_skeleton_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 로드
        videos = load_json_split(json_path)
        print(f"[INFO] Total videos: {len(videos)}")
        
        # 멀티프로세싱용 인자 준비
        process_args = [
            (video_info, output_skeleton_dir, args.model, args.conf, args.kpt_conf, 
             args.device, args.max_persons, normal_video_root, abnormal_video_root)
            for video_info in videos
        ]
        
        # 멀티프로세싱으로 병렬 처리
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        print(f"[INFO] Starting parallel extraction with {args.num_workers} workers...")
        
        with mp.Pool(processes=args.num_workers) as pool:
            # imap_unordered로 결과를 순서 상관없이 받음 (더 빠름)
            results = list(tqdm(
                pool.imap_unordered(process_single_video, process_args),
                total=len(process_args),
                desc=f"Extracting {dataset_name}"
            ))
        
        # 결과 집계
        for status, filename in results:
            if status == 'success':
                success_count += 1
            elif status == 'skip':
                skip_count += 1
            elif status == 'fail':
                fail_count += 1
        
        print(f"\n[RESULT] {dataset_name.upper()}")
        print(f"  Success: {success_count}")
        print(f"  Skipped (already exists): {skip_count}")
        print(f"  Failed (not found): {fail_count}")
        print(f"  Total processed: {success_count + skip_count}")
        print(f"  Output dir: {output_skeleton_dir}")
    
    print(f"\n{'='*80}")
    print("All done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
