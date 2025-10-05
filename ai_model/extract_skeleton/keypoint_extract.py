"""
논문과 동일한 방식의 관절점 추출 파이프라인
YOLOv8 (사람 탐지) + ByteTrack (추적) + HRNet-like pose estimation (COCO17 관절점)

요구사항:
1. 훈련: 정상 데이터만 사용
2. 테스트: 정상 데이터 일부 + 이상 데이터 (겹치지 않게)
3. 출력: 사람ID{프레임{관절좌표}} JSON 형식
4. 이상 데이터: theft_start~theft_end 구간을 1로 라벨링
"""

# OpenMP 라이브러리 충돌 문제 해결
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import os
from collections import defaultdict
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

class PoseLiftDatasetProcessor:
    def __init__(self):
        # YOLOv8 pose model (HRNet과 유사한 성능)
        self.model = YOLO('yolov8n-pose.pt')
        
        # COCO17 keypoint names (논문과 동일)
        self.keypoint_names = [
            'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
            'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
            'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
            'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
        ]
        
        self.normal_data_path = r"D:\AI-HUB_shoping"
        self.abnormal_data_path = r"D:\AI-HUB_shoplifting"
        
    def get_theft_frames(self, xml_path):
        """XML에서 theft_start와 theft_end 프레임 추출"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            theft_start = None
            theft_end = None
            
            for track in root.findall('.//track'):
                label = track.get('label', '')
                if label == 'theft_start':
                    # box 요소에서 frame 속성 찾기
                    box = track.find('.//box')
                    if box is not None:
                        theft_start = int(box.get('frame'))
                elif label == 'theft_end':
                    # box 요소에서 frame 속성 찾기
                    box = track.find('.//box')
                    if box is not None:
                        theft_end = int(box.get('frame'))
            
            return theft_start, theft_end
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return None, None
    
    def extract_keypoints_from_video(self, video_path, xml_path=None):
        """
        비디오에서 관절점 추출
        논문의 방법: YOLOv8 detection + ByteTrack tracking + Pose estimation
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 출력 데이터 구조: {person_id: {frame_num: {joint_name: [x, y]}}})
        pose_data = defaultdict(lambda: defaultdict(dict))
        frame_labels = np.zeros(total_frames)  # 0: normal, 1: abnormal
        
        # 이상 데이터인 경우 도난 프레임 구간 확인
        if xml_path and os.path.exists(xml_path):
            theft_start, theft_end = self.get_theft_frames(xml_path)
            if theft_start is not None and theft_end is not None:
                frame_labels[theft_start:theft_end+1] = 1
                print(f"도난 구간: {theft_start} ~ {theft_end} 프레임")
        
        frame_idx = 0
        print(f"처리 중: {os.path.basename(video_path)} ({total_frames} 프레임)")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # YOLOv8 pose estimation with tracking (ByteTrack 내장)
            results = self.model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                # 추적된 사람 ID들
                person_ids = results[0].boxes.id.int().cpu().tolist()
                # 각 사람의 관절점 좌표와 confidence (17개 keypoints)
                keypoints_xy = results[0].keypoints.xy.cpu().numpy()  # [N, 17, 2]
                keypoints_conf = results[0].keypoints.conf.cpu().numpy()  # [N, 17]
                
                for person_id, keypoints, confidences in zip(person_ids, keypoints_xy, keypoints_conf):
                    joint_data = {}
                    for i, ((x, y), conf) in enumerate(zip(keypoints, confidences)):
                        joint_name = self.keypoint_names[i]
                        # confidence와 함께 저장
                        if x > 0 and y > 0 and conf > 0.3:  # confidence threshold
                            joint_data[joint_name] = [float(x), float(y), float(conf)]
                        else:
                            joint_data[joint_name] = [0.0, 0.0, 0.0]  # 탐지 안된 경우
                    
                    pose_data[person_id][frame_idx] = joint_data
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  진행률: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
        
        cap.release()
        
        # defaultdict을 일반 dict로 변환
        pose_data = {k: dict(v) for k, v in pose_data.items()}
        
        return pose_data, frame_labels
    
    def apply_post_processing(self, pose_data):
        """
        논문의 후처리 적용:
        1. 선형 보간법으로 빠진 프레임 채우기
        2. 8-frame window 스무딩
        """
        processed_data = {}
        
        for person_id, frames_data in pose_data.items():
            if len(frames_data) < 2:  # 프레임이 너무 적으면 스킵
                continue
            
            frame_nums = sorted(frames_data.keys())
            processed_frames = {}
            
            # 1. 선형 보간법 적용
            for joint_name in self.keypoint_names:
                joint_positions = []
                valid_frames = []
                
                # 유효한 관절점 위치 수집
                for frame_num in frame_nums:
                    if joint_name in frames_data[frame_num]:
                        joint_data = frames_data[frame_num][joint_name]
                        if len(joint_data) >= 3:  # [x, y, confidence] 형태
                            x, y, conf = joint_data[0], joint_data[1], joint_data[2]
                            if x > 0 and y > 0 and conf > 0.3:  # 유효한 좌표
                                joint_positions.append([x, y, conf])
                                valid_frames.append(frame_num)
                        elif len(joint_data) >= 2:  # [x, y] 형태
                            x, y = joint_data[0], joint_data[1]
                            if x > 0 and y > 0:  # 유효한 좌표
                                joint_positions.append([x, y, 1.0])  # confidence 1.0으로 설정
                                valid_frames.append(frame_num)
                
                if len(valid_frames) < 2:
                    continue
                
                # 누락된 프레임들을 선형 보간으로 채우기
                for i, frame_num in enumerate(frame_nums):
                    if frame_num not in processed_frames:
                        processed_frames[frame_num] = {}
                    
                    if frame_num in valid_frames:
                        # 이미 유효한 데이터가 있는 경우
                        processed_frames[frame_num][joint_name] = frames_data[frame_num][joint_name]
                    else:
                        # 선형 보간 적용
                        # 가장 가까운 앞뒤 유효 프레임 찾기
                        prev_frame = None
                        next_frame = None
                        
                        for vf in valid_frames:
                            if vf < frame_num:
                                prev_frame = vf
                            elif vf > frame_num and next_frame is None:
                                next_frame = vf
                                break
                        
                        if prev_frame is not None and next_frame is not None:
                            # 선형 보간
                            alpha = (frame_num - prev_frame) / (next_frame - prev_frame)
                            prev_pos = frames_data[prev_frame][joint_name]
                            next_pos = frames_data[next_frame][joint_name]
                            
                            # confidence도 함께 보간
                            if len(prev_pos) >= 3 and len(next_pos) >= 3:
                                interp_x = prev_pos[0] + alpha * (next_pos[0] - prev_pos[0])
                                interp_y = prev_pos[1] + alpha * (next_pos[1] - prev_pos[1])
                                interp_conf = prev_pos[2] + alpha * (next_pos[2] - prev_pos[2])
                                processed_frames[frame_num][joint_name] = [interp_x, interp_y, interp_conf]
                            else:
                                interp_x = prev_pos[0] + alpha * (next_pos[0] - prev_pos[0])
                                interp_y = prev_pos[1] + alpha * (next_pos[1] - prev_pos[1])
                                processed_frames[frame_num][joint_name] = [interp_x, interp_y, 0.5]  # 기본 confidence
                        else:
                            processed_frames[frame_num][joint_name] = [0.0, 0.0, 0.0]
            
            # 2. 8-frame window 스무딩 적용
            smoothed_frames = {}
            window_size = 8
            half_window = window_size // 2
            
            for frame_num in frame_nums:
                smoothed_frames[frame_num] = {}
                
                for joint_name in self.keypoint_names:
                    positions = []
                    
                    # 윈도우 범위의 프레임들에서 위치 수집
                    for offset in range(-half_window, half_window + 1):
                        target_frame = frame_num + offset
                        if target_frame in processed_frames and joint_name in processed_frames[target_frame]:
                            pos = processed_frames[target_frame][joint_name]
                            if len(pos) >= 3 and pos[0] > 0 and pos[1] > 0 and pos[2] > 0.3:  # 유효한 위치만
                                positions.append(pos)
                    
                    if positions:
                        # 평균으로 스무딩
                        avg_x = sum(pos[0] for pos in positions) / len(positions)
                        avg_y = sum(pos[1] for pos in positions) / len(positions)
                        avg_conf = sum(pos[2] for pos in positions) / len(positions)
                        smoothed_frames[frame_num][joint_name] = [avg_x, avg_y, avg_conf]
                    else:
                        smoothed_frames[frame_num][joint_name] = [0.0, 0.0, 0.0]
            
            processed_data[person_id] = smoothed_frames
        
        return processed_data
    
    def save_results_to_json(self, results, output_path):
        """결과를 PoseLift 논문 형식의 JSON으로 저장"""
        # PoseLift 논문 형식: person_id가 최상위 키, frame이 두 번째 레벨
        formatted_results = {}
        
        # results는 person_id를 키로 하는 딕셔너리
        for person_id, frames_data in results.items():
            person_id_str = str(person_id)
            formatted_results[person_id_str] = {}
            
            # frames_data는 frame_num을 키로 하는 딕셔너리
            for frame_num, keypoints_dict in frames_data.items():
                # keypoints를 1차원 배열로 변환 (x,y,confidence 순서로 17개 관절)
                keypoints_array = []
                joint_order = [
                    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
                    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
                    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
                    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
                ]
                
                for joint_name in joint_order:
                    if joint_name in keypoints_dict:
                        coord = keypoints_dict[joint_name]
                        if len(coord) >= 3:
                            # 이미 [x, y, confidence] 형태
                            keypoints_array.extend([coord[0], coord[1], coord[2]])
                        else:
                            # [x, y] 형태인 경우 confidence를 1.0으로 설정
                            keypoints_array.extend([coord[0], coord[1], 1.0])
                    else:
                        # 관절이 없는 경우 0으로 채움
                        keypoints_array.extend([0.0, 0.0, 0.0])
                
                formatted_results[person_id_str][str(frame_num)] = {
                    'keypoints': keypoints_array,
                    'scores': None
                }
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
    
    def save_labels_to_npy(self, labels, output_file):
        """레이블을 npy 파일로 저장"""
        if labels is not None and len(labels) > 0:
            np.save(output_file, labels)
            print(f"Labels saved to: {output_file}")
        else:
            print("No labels to save")
    
    def process_dataset(self, output_dir="output"):
        """전체 데이터셋 처리"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 정상 데이터 처리 (훈련용)
        print("=== 정상 데이터 처리 (훈련용) ===")
        normal_train_dir = Path(self.normal_data_path) / "shoping_data" / "Training" / "raw_data"
        
        normal_videos = []
        for category_dir in normal_train_dir.iterdir():
            if category_dir.is_dir():
                for video_file in category_dir.glob("*.mp4"):
                    normal_videos.append(video_file)
        
        # 정상 데이터를 80% 훈련, 20% 테스트로 분할
        np.random.seed(42)
        normal_videos = list(normal_videos)
        np.random.shuffle(normal_videos)
        
        split_idx = int(len(normal_videos) * 0.8)
        train_videos = normal_videos[:split_idx]
        normal_test_videos = normal_videos[split_idx:]
        
        print(f"정상 데이터: 훈련 {len(train_videos)}개, 테스트 {len(normal_test_videos)}개")
        
    def process_single_video(self, video_path, xml_path=None):
        """단일 비디오 처리 (테스트용)"""
        print(f"Processing: {video_path}")
        
        # 관절점 추출
        pose_data, frame_labels = self.extract_keypoints_from_video(video_path, xml_path)
        
        if not pose_data:
            print("No pose data extracted")
            return None, None
        
        # 후처리 적용
        processed_data = self.apply_post_processing(pose_data)
        
        return processed_data, frame_labels

if __name__ == "__main__":
    # 테스트를 위해 단일 비디오 처리
    processor = PoseLiftDatasetProcessor()
    
    # 정상 데이터 테스트
    normal_video = r"D:\AI-HUB_shoping\shoping_data\Training\raw_data\TS_01.매장이동_01.매장이동_1\C_1_1_10_BU_DYA_08-13_14-41-55_CA_DF1_F1_F1.mp4"
    
    if os.path.exists(normal_video):
        print("=== 정상 데이터 테스트 ===")
        pose_data, labels = processor.process_single_video(normal_video)
        
        if pose_data:
            print(f"추출된 사람 수: {len(pose_data)}")
            for person_id, frames in pose_data.items():
                print(f"Person {person_id}: {len(frames)} 프레임")
        
        # PoseLift 형식으로 결과 저장
        processor.save_results_to_json(pose_data, "test_normal_result_poselift_format.json")
        
        # 레이블을 npy 파일로 저장
        processor.save_labels_to_npy(labels, "test_normal_labels.npy")
        
        print("PoseLift 형식 결과 저장: test_normal_result_poselift_format.json")
        print("레이블 저장: test_normal_labels.npy")
    else:
        print(f"비디오 파일을 찾을 수 없습니다: {normal_video}")
    
    # 이상 데이터 테스트
    abnormal_video = r"D:\AI-HUB_shoplifting\shoplift_data\Training\raw_data\Shoplift\C_3_12_10_BU_DYA_07-27_13-01-22_CA_RGB_DF2_F2.mp4"
    abnormal_xml = r"D:\AI-HUB_shoplifting\shoplift_data\Training\label_data\Shoplift\C_3_12_10_BU_DYA_07-27_13-01-22_CA_RGB_DF2_F2.xml"
    
    if os.path.exists(abnormal_video) and os.path.exists(abnormal_xml):
        print("\n=== 이상 데이터 테스트 ===")
        pose_data, labels = processor.process_single_video(abnormal_video, abnormal_xml)
        
        if pose_data:
            print(f"추출된 사람 수: {len(pose_data)}")
            for person_id, frames in pose_data.items():
                print(f"Person {person_id}: {len(frames)} 프레임")
            
            if labels is not None:
                abnormal_frames = np.sum(labels)
                print(f"이상 프레임 수: {abnormal_frames}/{len(labels)}")
        
        # PoseLift 형식으로 결과 저장
        processor.save_results_to_json(pose_data, "test_abnormal_result_poselift_format.json")
        
        # 레이블을 npy 파일로 저장
        processor.save_labels_to_npy(labels, "test_abnormal_labels.npy")
        
        print("PoseLift 형식 결과 저장: test_abnormal_result_poselift_format.json")
        print("레이블 저장: test_abnormal_labels.npy")
    else:
        print(f"이상 데이터 파일을 찾을 수 없습니다:")
        print(f"  Video: {abnormal_video}")
        print(f"  XML: {abnormal_xml}")
