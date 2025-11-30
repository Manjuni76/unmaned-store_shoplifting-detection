"""
데이터셋 관련 코드
ShopliftingDataset 클래스 및 데이터 로딩 유틸리티 함수
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

# Config import
from args import Config
import math


# 증강 변환 유틸리티
def get_aff_trans_mat(sx=1, sy=1, tx=0, ty=0, rot=0, shearx=0., sheary=0., flip=False):
    """
    Generate affine transfomation matrix (torch.tensor type) for transforming pose sequences
    :rot is given in degrees
    """
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_mat = torch.eye(3, dtype=torch.float32)
    if flip:
        flip_mat[0, 0] = -1.0
    trans_scale_mat = torch.tensor([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=torch.float32)
    shear_mat = torch.tensor([[1, shearx, 0], [sheary, 1, 0], [0, 0, 1]], dtype=torch.float32)
    rot_mat = torch.tensor([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=torch.float32)
    aff_mat = torch.matmul(rot_mat, trans_scale_mat)
    aff_mat = torch.matmul(shear_mat, aff_mat)
    aff_mat = torch.matmul(flip_mat, aff_mat)
    return aff_mat


def apply_pose_transform(pose, trans_mat):
    """
    Pose 시퀀스에 affine 변환 적용
    Shape: (Channels, Time_steps, Vertices, M)
    3 Channels: x, y, confidence
    """
    # confidence 벡터 분리 후 변환, 나중에 다시 결합
    conf = np.expand_dims(pose[2], axis=0)
    ones_vec = np.ones_like(conf)
    pose_w_ones = np.concatenate([pose[:2], ones_vec], axis=0)
    if len(pose.shape) == 3:
        einsum_str = 'ktv,ck->ctv'
    else:
        einsum_str = 'ktvm,ck->ctvm'
    pose_transformed_wo_conf = np.einsum(einsum_str, pose_w_ones, trans_mat)
    pose_transformed = np.concatenate([pose_transformed_wo_conf[:2], conf], axis=0)
    return pose_transformed


# 데이터 증강 변환 함수들
def transform_identity(x):
    """항등 변환"""
    return x


def transform_flip(x):
    """좌우 반전"""
    return apply_pose_transform(x, get_aff_trans_mat(flip=True))


def transform_shear(x):
    """전단 변환"""
    return apply_pose_transform(x, get_aff_trans_mat(shearx=0.1, sheary=0.1))


def transform_flip_shear(x):
    """좌우 반전 + 전단"""
    return apply_pose_transform(x, get_aff_trans_mat(flip=True, shearx=0.1, sheary=0.1))


# 데이터 전처리 함수들
def interpolate_skeleton(data):
    """
    스켈레톤 데이터의 결측치(0 값)를 선형 보간으로 채움
    
    Args:
        data: (T, V, C) 형태의 스켈레톤 데이터
    
    Returns:
        보간된 스켈레톤 데이터 (T, V, C)
    """
    T, V, C = data.shape
    interpolated_data = data.copy()
    
    for v in range(V):  # 각 관절에 대해
        for c in range(C):  # 각 채널(x, y, confidence)에 대해
            series = data[:, v, c]
            
            # 0을 NaN으로 변경 (보간 대상)
            series_with_nan = series.copy()
            series_with_nan[series == 0] = np.nan
            
            # Pandas의 interpolate 사용 (양방향 보간)
            s = pd.Series(series_with_nan)
            s = s.interpolate(method='linear', limit_direction='both')
            
            # NaN이 여전히 남아있으면(전체가 0인 경우) 0으로 채움
            interpolated_data[:, v, c] = s.fillna(0).values
    
    return interpolated_data


def convert_to_relative_coordinates(data, root_joint_idx=1):
    """
    절대 좌표를 상대 좌표로 변환 (Root-Relative Normalization)
    
    Args:
        data: (C, T, V) 형태의 스켈레톤 데이터
        root_joint_idx: 기준 관절 인덱스 (1=Neck, COCO18 기준)
    
    Returns:
        상대 좌표로 변환된 데이터 (C, T, V)
    """
    # data shape: (C, T, V)
    # C=0: x좌표, C=1: y좌표, C=2: confidence
    
    # 기준 관절(Neck)의 좌표 추출
    root_joint = data[:2, :, root_joint_idx:root_joint_idx+1]  # (2, T, 1) - x, y만
    
    # x, y 좌표를 상대 좌표로 변환 (confidence는 그대로 유지)
    data_relative = data.copy()
    data_relative[:2, :, :] = data[:2, :, :] - root_joint  # x, y에만 적용
    
    return data_relative


def apply_scale_normalization(data, neck_idx=1, hip_idx=8):
    """
    몸통 길이 기반 스케일 정규화
    세그먼트 전체의 중앙값을 사용하여 OpenPose 좌표 떨림 노이즈 방지
    
    Args:
        data: (C, T, V) 형태의 스켈레톤 데이터
        neck_idx: 목 관절 인덱스 (COCO18 기준 1)
        hip_idx: 골반 중심 관절 인덱스 (COCO18 기준 8)
    
    Returns:
        스케일 정규화된 데이터 (C, T, V)
    """
    # PyTorch Tensor로 변환
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = data.clone()
    
    C, T, V = data.shape
    
    # Neck과 Hip 좌표 추출 (x, y만 사용)
    neck_pos = data[:2, :, neck_idx]  # (2, T)
    hip_pos = data[:2, :, hip_idx]    # (2, T)
    
    # 몸통 길이 계산: sqrt((x_neck - x_hip)^2 + (y_neck - y_hip)^2)
    torso_len = torch.norm(neck_pos - hip_pos, dim=0)  # (T,)
    
    # 예외 처리: 길이가 너무 작거나 0인 경우 필터링
    valid_lens = torso_len[torso_len > 1e-3]
    
    if len(valid_lens) > 0:
        # 세그먼트 전체의 중앙값으로 통일 (프레임별 떨림 방지)
        median_len = torch.median(valid_lens)
        scale_factor = median_len.view(1, 1, 1)
    else:
        return data.numpy() if isinstance(data, torch.Tensor) else data
    
    # 스케일 정규화
    normalized_data = data / scale_factor
    
    return normalized_data.numpy()





# ShopliftingDataset 클래스
class ShopliftingDataset(Dataset):
    """
    폴더 기반 스켈레톤 데이터셋
    STG-NF 스타일 전처리 적용
    """
    def __init__(self, skeleton_base_path, seg_len=24, seg_stride=6, 
                 joint_subset=None, normalize=True, apply_augmentation=False, 
                 vid_res=[1920, 1080], use_cache=True, load_per_batch=False, 
                 preprocess_cache=True, filter_label=None):
        """
        Args:
            skeleton_base_path: 스켈레톤 데이터가 있는 기본 경로
            seg_len: 시퀀스 길이
            seg_stride: 시퀀스 stride
            joint_subset: 사용할 관절 인덱스 (None이면 전체)
            normalize: pose normalization 여부 (STG-NF 방식)
            apply_augmentation: 데이터 증강 여부
            vid_res: 비디오 해상도 [width, height] for normalization
            use_cache: 스켈레톤 데이터 캐싱 여부 (메모리 vs 속도 트레이드오프)
            load_per_batch: True이면 배치마다 스켈레톤 로드 (메모리 절약), False이면 사전 생성
            preprocess_cache: True이면 전처리된 세그먼트를 메모리에 캐싱 (매우 빠름, 메모리 많이 사용)
            filter_label: 'normal' 또는 'abnormal'로 특정 라벨만 필터링 (None이면 전체)
        """
        self.seg_len = seg_len
        self.seg_stride = seg_stride
        self.joint_subset = joint_subset
        self.normalize = normalize
        self.skeleton_base_path = skeleton_base_path
        self.load_per_batch = load_per_batch
        self.apply_augmentation = apply_augmentation
        self.vid_res = vid_res
        self.use_cache = use_cache
        self.preprocess_cache = preprocess_cache
        
        # filter_label 저장 (나중에 세그먼트 필터링에 사용)
        self.filter_label = filter_label
        
        # 정상/이상 데이터 통합 (영상 레벨)
        self.samples = []
        
        # 폴더 직접 스캔
        import glob
        skeleton_files = glob.glob(os.path.join(skeleton_base_path, '*.json'))
        for skeleton_file in skeleton_files:
            filename = os.path.basename(skeleton_file)
            item = {
                'filename': filename,
                'label': 0  # 기본값 정상
            }
            self.samples.append(item)
        
        print(f"총 {len(self.samples)}개 영상 로드")
        
        # 전체 학습 데이터 통계 로드/생성 (일관된 정규화 위해)
        stats_file = os.path.join(os.path.dirname(__file__), "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.global_mean_x = stats["global_mean_x"]
                self.global_mean_y = stats["global_mean_y"]
                self.global_std_y = stats["global_std_y"]
                print(f"전역 통계 로드됨 (mean_x={self.global_mean_x:.4f}, mean_y={self.global_mean_y:.4f}, std_y={self.global_std_y:.4f})")
        else:
            # 통계 파일이 없으면 자동 생성
            print("dataset_stats.json 파일 없음. 통계 계산 중...")
            self._compute_and_save_stats(stats_file)
        
        # 스켈레톤 데이터 캐시
        self.skeleton_cache = {} if use_cache else None
        
        # 전처리된 세그먼트 캐시 (augmentation 제외)
        self.preprocessed_cache = {} if preprocess_cache and not apply_augmentation else None
        
        # 데이터 증강 변환 리스트
        if self.apply_augmentation:
            self.transform_list = [
                transform_identity,      # Identity
                transform_flip,          # Flip
                transform_shear,         # Shear
                transform_flip_shear,    # Flip + Shear
            ]
        else:
            self.transform_list = [transform_identity]
        
        # filter_label 저장 (나중에 세그먼트 필터링에 사용)
        self.filter_label = filter_label
        
        # 스켈레톤 시퀀스 생성
        self.segments = []
        self.segment_labels = []
        self.segment_metadata = []
        
        if not self.load_per_batch:
            # 사전에 모든 세그먼트 생성
            self._generate_segments()
        else:
            # 배치마다 로드하기 위한 메타데이터만 생성
            self._generate_segment_metadata()
    
    def _compute_and_save_stats(self, stats_file):
        """
        전체 학습 데이터 전역 통계(mean, std) 계산 및 JSON 저장
        train/test 일관된 정규화를 위해 사용
        """
        print("통계 계산 시작")
        
        all_x = []
        all_y = []
        total_frames = 0
        
        # 모든 스켈레톤 데이터 로드
        for sample in tqdm(self.samples, desc="스켈레톤 데이터 로드 중"):
            filename = sample['filename'].replace('.mp4', '.json')
            skeleton_path = os.path.join(self.skeleton_base_path, filename)
            
            if not os.path.exists(skeleton_path):
                continue
            
            pose_data = self._load_skeleton(skeleton_path)
            if pose_data is None:
                continue
            
            # 프레임 수 카운트
            total_frames += pose_data.shape[0]
            
            # 픽셀 좌표 → [0, 1] 정규화
            vid_res_wconf = np.array(self.vid_res + [1], dtype=np.float32)
            pose_normalized = pose_data / vid_res_wconf
            
            # SYMM_RANGE 적용 ([-1, 1])
            if Config.Data.SYMM_RANGE:
                pose_normalized[..., :2] = 2 * pose_normalized[..., :2] - 1
            
            # x, y 좌표만 추출 (confidence 제외)
            x_coords = pose_normalized[:, :, 0].flatten()  # 모든 x 좌표
            y_coords = pose_normalized[:, :, 1].flatten()  # 모든 y 좌표
            
            # 0이 아닌 유효한 값만 수집 (결측치 제외)
            valid_x = x_coords[x_coords != 0]
            valid_y = y_coords[y_coords != 0]
            
            all_x.extend(valid_x.tolist())
            all_y.extend(valid_y.tolist())
        
        # NumPy 배열로 변환
        all_x = np.array(all_x, dtype=np.float32)
        all_y = np.array(all_y, dtype=np.float32)
        
        # 통계 계산
        self.global_mean_x = float(np.mean(all_x))
        self.global_mean_y = float(np.mean(all_y))
        self.global_std_y = float(np.std(all_y))
        
        # JSON 파일로 저장
        stats = {
            "global_mean_x": self.global_mean_x,
            "global_mean_y": self.global_mean_y,
            "global_std_y": self.global_std_y,
            "num_frames": total_frames,
            "resolution": self.vid_res
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"통계 계산 완료. 파일 저장됨: {stats_file}")
        print(f"mean_x={self.global_mean_x:.4f}, mean_y={self.global_mean_y:.4f}, std_y={self.global_std_y:.4f}")
        print(f"총 {total_frames:,}개 프레임 분석됨 (해상도: {self.vid_res})")
    
    def _generate_segment_metadata(self):
        """배치별 로딩을 위한 메타데이터 생성"""
        for sample in tqdm(self.samples, desc="Generating metadata"):
            filename = sample['filename'].replace('.mp4', '.json')
            skeleton_path = os.path.join(self.skeleton_base_path, filename)
            
            if not os.path.exists(skeleton_path):
                print(f"Warning: 스켈레톤 파일 없음 - {skeleton_path}")
                continue
            
            # 스켈레톤 데이터 로드하여 프레임 수 확인
            pose_data = self._load_skeleton(skeleton_path)
            if pose_data is None:
                continue
            
            T = pose_data.shape[0]
            
            # 세그먼트 메타데이터 생성
            for start_idx in range(0, T - self.seg_len + 1, self.seg_stride):
                center_frame_idx = start_idx + self.seg_len // 2
                
                # 라벨 계산
                if sample['label'] == 1:  # 이상 데이터
                    theft_start = sample.get('theft_start', 0)
                    theft_end = sample.get('theft_end', T)
                    label = 1 if theft_start <= center_frame_idx <= theft_end else 0
                else:
                    label = 0
                
                self.segment_metadata.append({
                    'skeleton_path': skeleton_path,
                    'start_idx': start_idx,
                    'center_frame': center_frame_idx,
                    'label': label,
                    'filename': sample['filename'],
                    'video_label': sample['label']
                })
                self.segment_labels.append(label)  # 레이블 수집
        
        print(f"총 {len(self.segment_metadata)}개 세그먼트 메타데이터 생성됨")
        normal_count = sum(1 for m in self.segment_metadata if m['label'] == 0)
        abnormal_count = sum(1 for m in self.segment_metadata if m['label'] == 1)
        print(f"정상: {normal_count}개, 이상: {abnormal_count}개")
    
    def _load_skeleton(self, skeleton_path):
        """스켈레톤 JSON 파일 로드 (STG-NF 포맷 변환) - 캐싱 지원"""
        # 캐시 확인
        if self.use_cache and skeleton_path in self.skeleton_cache:
            return self.skeleton_cache[skeleton_path]
        
        try:
            with open(skeleton_path, 'r', encoding='utf-8') as f:
                skeleton_data = json.load(f)
            
            # person_1의 데이터만 사용 (단일 person 가정)
            if 'person_1' not in skeleton_data:
                print(f"Warning: person_1 데이터 없음 - {skeleton_path}")
                return None
            
            person_data = skeleton_data['person_1']
            
            # 프레임 ID를 정수로 변환하여 정렬
            frame_ids = sorted([int(fid) for fid in person_data.keys()])
            
            frames = []
            for frame_id in frame_ids:
                frame_str = str(frame_id)
                if frame_str in person_data:
                    frame_info = person_data[frame_str]
                    if 'keypoints' in frame_info:
                        keypoints = np.array(frame_info['keypoints'], dtype=np.float32)  # (V, 3)
                        frames.append(keypoints)
            
            if len(frames) == 0:
                return None
            
            pose_data = np.stack(frames, axis=0)  # (T, V, 3)
            
            # COCO17 -> COCO18 변환 (필요시)
            if pose_data.shape[1] == 17:
                pose_data = self._keypoints17_to_coco18(pose_data)
            
            # FPS 통일: C_1로 시작하는 10fps 영상을 3fps로 다운샘플링
            filename = os.path.basename(skeleton_path)
            
            # C_1로 시작하는 파일은 10fps 영상
            if filename.startswith('C_1_'):
                # 10fps -> 3fps 변환 (stride=3)
                pose_data = pose_data[::3]
                print(f"FPS 다운샘플링 (10fps->3fps): {filename} ({len(frames)}->{pose_data.shape[0]} frames)")
            
            # 캐시에 저장
            if self.use_cache:
                self.skeleton_cache[skeleton_path] = pose_data
            
            return pose_data
        except Exception as e:
            print(f"Error: 스켈레톤 로드 실패 - {skeleton_path}: {e}")
            return None
    
    def _keypoints17_to_coco18(self, kps):
        """
        Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
        New keypoint (neck) is the average of the shoulders, and points are also reordered.
        """
        kp_np = np.array(kps, dtype=np.float32)
        neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
        kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
        opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        opp_order = np.array(opp_order, dtype=int)
        kp_coco18 = kp_np[..., opp_order, :]
        return kp_coco18.astype(np.float32)
    
    def _generate_segments(self):
        """
        영상별 시퀀스 세그먼트 생성
        중앙 프레임 라벨을 세그먼트 라벨로 사용
        """
        for sample in tqdm(self.samples, desc="Generating segments"):
            # 스켈레톤 파일 경로
            filename = sample['filename'].replace('.mp4', '.json')
            skeleton_path = os.path.join(self.skeleton_base_path, filename)
            
            if not os.path.exists(skeleton_path):
                print(f"Warning: 스켈레톤 파일 없음 - {skeleton_path}")
                continue
            
            # 스켈레톤 데이터 로드 (T, V, 3)
            pose_data = self._load_skeleton(skeleton_path)
            if pose_data is None:
                continue
            
            T, V, C = pose_data.shape  # (frames, 18, 3)
            
            # 시퀀스 세그먼트 생성
            for start_idx in range(0, T - self.seg_len + 1, self.seg_stride):
                segment = pose_data[start_idx:start_idx + self.seg_len, :, :]  # (seg_len, V, 3)
                
                # 유효성 검사: 모든 값이 0인 프레임이 너무 많으면 제외
                frame_sums = np.abs(segment).sum(axis=(1, 2))  # (seg_len,)
                valid_frames = (frame_sums > 1e-5).sum()
                
                # 유효 프레임이 절반 미만이면 스킵
                if valid_frames < (self.seg_len // 2):
                    continue
                
                # 중앙 프레임 인덱스
                center_frame_idx = start_idx + self.seg_len // 2
                
                # 중앙 프레임의 GT 라벨 사용
                if sample['label'] == 1:  # 이상 데이터
                    theft_start = sample.get('theft_start', 0)
                    theft_end = sample.get('theft_end', T)
                    # 중앙 프레임이 이상 구간에 있으면 1
                    if theft_start <= center_frame_idx <= theft_end:
                        label = 1
                    else:
                        label = 0
                else:
                    label = 0
                
                self.segments.append(segment)
                self.segment_labels.append(label)
                self.segment_metadata.append({
                    'filename': sample['filename'],
                    'start_frame': start_idx,
                    'center_frame': center_frame_idx,
                    'video_label': sample['label']
                })
        
        # filter_label에 따라 세그먼트 필터링
        if self.filter_label is not None:
            filtered_segments = []
            filtered_labels = []
            filtered_metadata = []
            
            target_label = 0 if self.filter_label.lower() == 'normal' else 1
            
            for seg, lbl, meta in zip(self.segments, self.segment_labels, self.segment_metadata):
                if lbl == target_label:
                    filtered_segments.append(seg)
                    filtered_labels.append(lbl)
                    filtered_metadata.append(meta)
            
            self.segments = filtered_segments
            self.segment_labels = filtered_labels
            self.segment_metadata = filtered_metadata
            
            print(f"필터링 ({self.filter_label}): {len(self.segments)}개 세그먼트")
        
        print(f"총 {len(self.segments)}개 세그먼트 생성됨")
        print(f"- 정상: {sum([1 for l in self.segment_labels if l == 0])}개")
        print(f"- 이상: {sum([1 for l in self.segment_labels if l == 1])}개")
    
    def __len__(self):
        if self.load_per_batch:
            return len(self.segment_metadata)
        else:
            return len(self.segments)
    
    def __getitem__(self, idx):
        """
        전처리 파이프라인 적용
        Returns: (C, T, V) 형식의 텐서
        """
        # 전처리 캐시 확인 (augmentation이 없을 때만)
        if self.preprocessed_cache is not None and idx in self.preprocessed_cache:
            segment_tensor, label_tensor = self.preprocessed_cache[idx]
            return segment_tensor, label_tensor
        
        if self.load_per_batch:
            # 배치별 로딩: 메타데이터에서 세그먼트 로드
            meta = self.segment_metadata[idx]
            pose_data = self._load_skeleton(meta['skeleton_path'])
            
            if pose_data is None:
                # 로드 실패 시 더미 데이터 반환
                V = len(self.joint_subset) if self.joint_subset else 18
                segment = np.zeros((self.seg_len, V, 3), dtype=np.float32)
                label = meta['label']
            else:
                # 세그먼트 추출
                start_idx = meta['start_idx']
                segment = pose_data[start_idx:start_idx + self.seg_len, :, :]
                label = meta['label']
        else:
            # 사전 생성 방식
            segment = np.array(self.segments[idx])  # (T, V, 3)
            label = self.segment_labels[idx]
        
        # 전처리 파이프라인
        
        # 0. 픽셀 좌표 → [-1, 1] 범위로 정규화
        vid_res_wconf = np.array(self.vid_res + [1], dtype=np.float32)  # [1920, 1080, 1]
        segment = segment / vid_res_wconf  # -> [0, 1]
        if Config.Data.SYMM_RANGE:
            segment[..., :2] = 2 * segment[..., :2] - 1  # x, y만 [-1, 1]
        
        # 1. 선형 보간: 결측치(0 값) 채우기
        segment = interpolate_skeleton(segment)  # (T, V, 3)
        
        # 2. (T, V, 3) -> (3, T, V) 변환
        segment_ctv = segment.transpose(2, 0, 1)  # (C=3, T, V=18)
        
        # 3. 부위 선택 먼저 수행
        if self.joint_subset is not None:
            segment_ctv = segment_ctv[:, :, self.joint_subset]  # (C, T, subset_V)
        
        # 4. x, y만 사용 (confidence 제거)
        segment_xy = segment_ctv[:2, :, :]  # (C=2, T, V)
        
        # 5. 중심화: 각 프레임에서 평균을 빼서 중심을 (0,0)으로
        segment_mean = segment_xy.mean(axis=2, keepdims=True)  # (2, T, 1)
        segment_xy = segment_xy - segment_mean
        # 6. 스케일 정규화: 전체 세그먼트의 표준편차로 나눔
        std_val = segment_xy.std()
        if std_val > 1e-6:
            segment_xy = segment_xy / std_val
        
        # 7. Augmentation 적용
        if self.apply_augmentation and random.random() < 0.5:
            transform = random.choice(self.transform_list)
            segment_xy = transform(segment_xy)
        
        segment_tensor = torch.FloatTensor(segment_xy)
        label_tensor = torch.LongTensor([label])[0]
        
        # 전처리 캐시에 저장 (augmentation이 없을 때만)
        if self.preprocessed_cache is not None:
            self.preprocessed_cache[idx] = (segment_tensor, label_tensor)
        
        return segment_tensor, label_tensor
    
    def get_metadata(self, idx):
        """세그먼트 메타데이터 반환"""
        return self.segment_metadata[idx]


# 데이터 로더 생성 함수
def create_dataloader(skeleton_base_path, seg_len, seg_stride, 
                      joint_subset=None, batch_size=64, shuffle=True, 
                      num_workers=12, normalize=True, apply_augmentation=False, 
                      vid_res=[1920, 1080]):
    """
    DataLoader 생성 헬퍼 함수
    """
    dataset = ShopliftingDataset(
        skeleton_base_path=skeleton_base_path,
        seg_len=seg_len,
        seg_stride=seg_stride,
        joint_subset=joint_subset,
        normalize=normalize,
        apply_augmentation=apply_augmentation,
        vid_res=vid_res
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset
