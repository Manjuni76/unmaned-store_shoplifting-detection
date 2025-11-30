import torch
import numpy as np
import os
import sys
import json
import tempfile
import cv2
from collections import defaultdict

# ai_model 경로 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_MODEL_DIR = os.path.join(BASE_DIR, 'ai_model')
sys.path.insert(0, AI_MODEL_DIR)

# ai_model에서 import (eval_pipeline.py와 동일)
from args import Config
from models.stgnf_loader import load_all_stgnf_models
from models.model_builder import create_attention_classifier

# skeleton 추출용
from extract_skeleton.skeleton_extract import SkeletonExtractorWithXML

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShopliftingDetector:
    """
    도난 탐지 전체 파이프라인
    1. 영상 → skeleton 추출
    2. 12프레임 이상 사람 있는 구간만 선별
    3. STG-NF + Attention 모델로 추론
    4. 도난 구간 반환
    """
    
    def __init__(self):
        print(f"AI Server 모델 로딩 시작... (Device: {DEVICE})")
        
        # 1. Skeleton 추출기 초기화
        yolo_model_path = os.path.join(AI_MODEL_DIR, 'extract_skeleton', 'yolo11x-pose.pt')
        self.skeleton_extractor = SkeletonExtractorWithXML(
            model_name=yolo_model_path,
            conf_threshold=0.5,
            kpt_threshold=0.3,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 2. STG-NF 모델 로드 (부위별 5개)
        print("[AI Server] STG-NF 모델 로딩...")
        self.stgnf_models = load_all_stgnf_models(DEVICE)
        
        # 3. Attention Classifier 로드
        print("[AI Server] Attention Classifier 로딩...")
        
        # 샘플 데이터 생성 (모델 구조 초기화용)
        sample_data_dict = {}
        for part in Config.Joint.BODY_PARTS:
            subset = Config.Joint.JOINT_SUBSET_MAP[part]
            num_joints = len(subset) if subset is not None else 18
            sample_data_dict[part] = torch.zeros(
                1, Config.STGNF.IN_CHANNELS, Config.Data.SEG_LEN, num_joints
            ).to(DEVICE)
        
        # Attention 모델 생성
        self.attention_model = create_attention_classifier(
            stg_nf_models_dict=self.stgnf_models,
            sample_data_dict=sample_data_dict,
            num_classes=Config.Attention.NUM_CLASSES,
            embed_dim=Config.Attention.EMBED_DIM,
            num_heads=Config.Attention.NUM_HEADS,
            num_encoder_layers=Config.Attention.NUM_ENCODER_LAYERS,
            dropout=Config.Attention.DROPOUT,
            device=str(DEVICE)
        )
        
        # 학습된 가중치 로드
        checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, 'attention_fin.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Attention 체크포인트 없음: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.attention_model.load_state_dict(checkpoint['model_state_dict'])
        self.attention_model.eval()
        
        print(f"[AI Server] 모델 로딩 완료!")
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - Val F1: {checkpoint.get('val_f1', 0):.2f}%")
        
    
    def extract_skeleton_from_video(self, video_path):
        print(f"AI Server Skeleton 추출 시작: {video_path}")
        
        # 임시 출력 폴더
        temp_output_dir = tempfile.mkdtemp()
        output_json_path = os.path.join(temp_output_dir, 'skeleton.json')
        
        try:
            # skeleton_extract.py의 extract_simple 메서드 사용
            success = self.skeleton_extractor.extract_simple(
                video_path=video_path,
                output_json_path=output_json_path
            )
            
            if not success:
                print(f"AI Server Skeleton 추출 실패")
                return None
            
            with open(output_json_path, 'r') as f:
                skeleton_data = json.load(f)
            
            # 프레임 수 계산
            person_data = skeleton_data.get('person_1', {})
            frame_count = len(person_data)
            
            print(f"[AI Server] Skeleton 추출 완료: {frame_count} 프레임")
            return skeleton_data
            
        except Exception as e:
            print(f"[AI Server] Skeleton 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # 임시 파일 정리
            import shutil
            shutil.rmtree(temp_output_dir, ignore_errors=True)
    
    
    def filter_valid_segments(self, skeleton_data, min_frames=12):
        person_data = skeleton_data.get('person_1', {})
        if not person_data:
            return []
        
        # 프레임 인덱스를 정수로 정렬
        frame_indices = sorted([int(k) for k in person_data.keys()])
        
        valid_segments = []
        current_start = None
        consecutive_count = 0
        
        for frame_idx in frame_indices:
            frame_data = person_data[str(frame_idx)]
            keypoints = frame_data.get('keypoints', [])
            
            # 사람이 있는지 확인 (keypoints가 모두 0이 아닌지)
            has_person = any(kp[0] != 0 or kp[1] != 0 for kp in keypoints)
            
            if has_person:
                if current_start is None:
                    current_start = frame_idx
                consecutive_count += 1
            else:
                if consecutive_count >= min_frames:
                    valid_segments.append((current_start, frame_idx - 1))
                current_start = None
                consecutive_count = 0
        
        # 마지막 구간 처리
        if consecutive_count >= min_frames:
            valid_segments.append((current_start, frame_indices[-1]))
        
        print(f"[AI Server] 유효 구간 {len(valid_segments)}개 발견 (최소 {min_frames}프레임 이상)")
        return valid_segments
    
    
    def preprocess_skeleton_to_tensor(self, skeleton_data, start_frame, seg_len=24):
        """
        skeleton_data에서 특정 구간(seg_len)을 텐서로 변환
        
        Args:
            skeleton_data: {"person_1": {"0": {"keypoints": [...]}, ...}}
            start_frame: 시작 프레임 인덱스
            seg_len: 세그먼트 길이
        
        Returns:
            torch.Tensor: (C=2, T=24, V=18) - x, y 좌표만
        """
        person_data = skeleton_data.get('person_1', {})
        
        # (T, V, C) 형태로 변환
        skeleton_array = np.zeros((seg_len, 18, 2), dtype=np.float32)
        
        for t in range(seg_len):
            frame_idx = start_frame + t
            frame_data = person_data.get(str(frame_idx), {})
            keypoints = frame_data.get('keypoints', [])
            
            if len(keypoints) >= 17:
                for v in range(17):
                    if v < len(keypoints):
                        x, y, conf = keypoints[v]
                        skeleton_array[t, v, 0] = x
                        skeleton_array[t, v, 1] = y
        
        # 정규화 (Config.Data.NORMALIZE=True인 경우)
        if Config.Data.NORMALIZE:
            # [-1, 1] 범위로 정규화
            skeleton_array[:, :, 0] /= Config.Data.VID_RES[0]  
            skeleton_array[:, :, 1] /= Config.Data.VID_RES[1]  
            skeleton_array = skeleton_array * 2 - 1  
        
        # (C, T, V) 형태로 변환
        tensor = torch.from_numpy(skeleton_array).permute(2, 0, 1)  # (2, 24, 18)
        return tensor
    
    
    @torch.no_grad()
    def predict(self, video_path):
        """
        전체 파이프라인 실행
        
        Returns:
            dict: {
                'is_abnormal': bool,
                'start_time_sec': float,
                'result_text': str (JSON 형태의 초 리스트)
            }
        """
        print(f"[AI Server] 분석 시작: {video_path}")
        
        # 1. Skeleton 추출
        skeleton_data = self.extract_skeleton_from_video(video_path)
        if skeleton_data is None:
            return {
                'is_abnormal': False,
                'start_time_sec': 0.0,
                'result_text': '[]'
            }
        
        # 2. 유효 구간 선별 (12프레임 이상)
        valid_segments = self.filter_valid_segments(skeleton_data, min_frames=12)
        if not valid_segments:
            print("[AI Server] 유효 구간 없음 (12프레임 이상 사람 없음)")
            return {
                'is_abnormal': False,
                'start_time_sec': 0.0,
                'result_text': '[]'
            }
        
        # 3. 각 세그먼트(24프레임)별로 모델 추론
        seg_len = Config.Data.SEG_LEN  # 24
        frame_scores = {}  # {frame_idx: score}
        
        for seg_start, seg_end in valid_segments:
            # 슬라이딩 윈도우 (stride=1)
            for start_frame in range(seg_start, seg_end - seg_len + 2):
                if start_frame + seg_len > seg_end + 1:
                    break
                
                # 텐서 변환
                x = self.preprocess_skeleton_to_tensor(skeleton_data, start_frame, seg_len)
                x = x.unsqueeze(0).to(DEVICE)  # (1, C, T, V)
                
                # 부위별 슬라이싱
                x_dict = {}
                for part in self.attention_model.part_names:
                    subset = Config.Joint.JOINT_SUBSET_MAP[part]
                    if subset is not None:
                        x_dict[part] = x[:, :, :, subset]
                    else:
                        x_dict[part] = x
                
                # Forward pass
                logits = self.attention_model(x_dict)
                probs = torch.softmax(logits, dim=1)
                abnormal_score = probs[0, 1].item() 
                
                # 프레임별 점수 저장 (max)
                for frame_idx in range(start_frame, start_frame + seg_len):
                    if frame_idx not in frame_scores:
                        frame_scores[frame_idx] = abnormal_score
                    else:
                        frame_scores[frame_idx] = max(frame_scores[frame_idx], abnormal_score)
        
        # 4. 위험도 평가 (확률 기반 4단계)
        # 최고 점수를 기준으로 전체 영상의 위험도 판정
        max_score = max(frame_scores.values()) if frame_scores else 0.0
        
        # 위험도 등급 분류
        if max_score < 0.80:
            risk_level = "정상"
            risk_color = "success"  # 녹색
            is_abnormal = False
        elif max_score < 0.87:
            risk_level = "주의 필요"
            risk_color = "warning"  # 노란색
            is_abnormal = False
        elif max_score < 0.98:
            risk_level = "도난 의심"
            risk_color = "warning"  # 주황색
            is_abnormal = True
        else:  # >= 0.98
            risk_level = "확실한 도난"
            risk_color = "danger"  # 빨간색
            is_abnormal = True
        
        # 높은 점수 프레임만 표시 (0.80 이상)
        high_risk_frames = {idx: score for idx, score in frame_scores.items() if score >= 0.80}
        
        # FPS 계산
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        # 연속된 프레임을 구간으로 묶기
        risk_segments = []
        if high_risk_frames:
            sorted_frames = sorted(high_risk_frames.items())
            
            # 구간 시작
            segment_start_frame = sorted_frames[0][0]
            segment_start_time = round(segment_start_frame / fps, 1)
            segment_max_score = sorted_frames[0][1]
            prev_frame = sorted_frames[0][0]
            
            for frame_idx, score in sorted_frames[1:]:
                # 연속된 프레임인지 확인 (1초 이내, 약 30프레임)
                if frame_idx - prev_frame <= fps:
                    # 구간 연장, 최고 점수 업데이트
                    segment_max_score = max(segment_max_score, score)
                else:
                    # 구간 종료, 저장
                    segment_end_time = round(prev_frame / fps, 1)
                    risk_segments.append({
                        "start_time": segment_start_time,
                        "end_time": segment_end_time,
                        "max_score": round(segment_max_score, 3)
                    })
                    
                    # 새 구간 시작
                    segment_start_frame = frame_idx
                    segment_start_time = round(frame_idx / fps, 1)
                    segment_max_score = score
                
                prev_frame = frame_idx
            
            # 마지막 구간 저장
            segment_end_time = round(prev_frame / fps, 1)
            risk_segments.append({
                "start_time": segment_start_time,
                "end_time": segment_end_time,
                "max_score": round(segment_max_score, 3)
            })
        
        start_time_sec = risk_segments[0]["start_time"] if risk_segments else 0.0
        
        print(f"[AI Server] 분석 완료: {risk_level} (최고 점수: {max_score:.3f}, 구간: {len(risk_segments)}개)")
        
        return {
            'is_abnormal': is_abnormal,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'max_score': round(max_score, 3),
            'start_time_sec': start_time_sec,
            'result_text': json.dumps(risk_segments) 
        }


# 전역 객체 생성 (서버 시작 시 한 번만 로딩)
print("[AI Server] Detector 초기화 중...")
detector = ShopliftingDetector()
print("[AI Server] Detector 준비 완료!")