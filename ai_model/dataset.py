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

# 전처리 함수 import
try:
    from utils.data_utils import normalize_pose, apply_pose_transform, get_aff_trans_mat
except ImportError:
    print("[WARNING] utils.data_utils import 실패 - 전처리 함수 사용 불가")
    normalize_pose = None
    apply_pose_transform = None
    get_aff_trans_mat = None


# ============================================================================
# Pickle 가능한 변환 함수들 (lambda 대신 일반 함수 사용)
# ============================================================================
def transform_identity(x):
    """항등 변환"""
    return x


def transform_flip(x):
    """좌우 반전 변환"""
    if apply_pose_transform is None or get_aff_trans_mat is None:
        return x
    return apply_pose_transform(x, get_aff_trans_mat(flip=True))


def transform_shear(x):
    """전단 변환"""
    if apply_pose_transform is None or get_aff_trans_mat is None:
        return x
    return apply_pose_transform(x, get_aff_trans_mat(shearx=0.1, sheary=0.1))


def transform_flip_shear(x):
    """좌우 반전 + 전단 변환"""
    if apply_pose_transform is None or get_aff_trans_mat is None:
        return x
    return apply_pose_transform(x, get_aff_trans_mat(flip=True, shearx=0.1, sheary=0.1))


# ============================================================================
# ShopliftingDataset 클래스
# ============================================================================
class ShopliftingDataset(Dataset):
    """
    data_split의 JSON 파일을 읽어서 스켈레톤 데이터를 로드하는 Dataset
    STG-NF 스타일의 전처리 방식 적용
    """
    def __init__(self, json_path, skeleton_base_path, seg_len=24, seg_stride=6, 
                 joint_subset=None, normalize=True, apply_augmentation=False, 
                 vid_res=[1920, 1080], use_cache=True, load_per_batch=False, 
                 preprocess_cache=True, filter_label=None):
        """
        Args:
            json_path: train_data.json, mlp_train_data.json, test_data.json 경로
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
        
        # 스켈레톤 데이터 캐시 (메모리에 저장)
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
        
        # JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # filter_label 저장 (나중에 세그먼트 필터링에 사용)
        self.filter_label = filter_label
        
        # 정상/이상 데이터 통합 (영상 레벨)
        self.samples = []
        
        # 영상 레벨에서는 모든 데이터 로드 (세그먼트 레벨에서 필터링)
        for item in self.data.get('normal', []):
            item['label'] = 0  # 정상
            self.samples.append(item)
        
        for item in self.data.get('abnormal', []):
            item['label'] = 1  # 이상
            self.samples.append(item)
        
        print(f"[DATASET] 총 {len(self.samples)}개 영상 로드")
        
        # 스켈레톤 시퀀스 생성
        self.segments = []
        self.segment_labels = []
        self.segment_metadata = []
        
        if not self.load_per_batch:
            # 사전에 모든 세그먼트 생성 (기존 방식)
            self._generate_segments()
        else:
            # 배치마다 로드하기 위한 메타데이터만 생성
            self._generate_segment_metadata()
    
    def _generate_segment_metadata(self):
        """배치별 로딩을 위한 메타데이터만 생성"""
        for sample in tqdm(self.samples, desc="Generating metadata"):
            filename = sample['filename'].replace('.mp4', '_skeleton.json')
            skeleton_path = os.path.join(self.skeleton_base_path, filename)
            
            if not os.path.exists(skeleton_path):
                print(f"[WARNING] 스켈레톤 파일 없음: {skeleton_path}")
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
        
        print(f"[DATASET] 총 {len(self.segment_metadata)}개 세그먼트 메타데이터 생성")
        normal_count = sum(1 for m in self.segment_metadata if m['label'] == 0)
        abnormal_count = sum(1 for m in self.segment_metadata if m['label'] == 1)
        print(f"[DATASET] 정상: {normal_count}개, 이상: {abnormal_count}개")
    
    def _load_skeleton(self, skeleton_path):
        """스켈레톤 JSON 파일 로드 (STG-NF 포맷으로 변환) - 캐싱 지원"""
        # 캐시 확인
        if self.use_cache and skeleton_path in self.skeleton_cache:
            return self.skeleton_cache[skeleton_path]
        
        try:
            with open(skeleton_path, 'r', encoding='utf-8') as f:
                skeleton_data = json.load(f)
            
            # person_1의 데이터만 사용 (단일 person 가정)
            if 'person_1' not in skeleton_data:
                print(f"[WARNING] person_1 없음: {skeleton_path}")
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
            
            # 캐시에 저장
            if self.use_cache:
                self.skeleton_cache[skeleton_path] = pose_data
            
            return pose_data
        except Exception as e:
            print(f"[ERROR] 스켈레톤 로드 실패 {skeleton_path}: {e}")
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
        영상별로 시퀀스 세그먼트 생성
        중앙 프레임의 라벨을 세그먼트 라벨로 사용
        """
        for sample in tqdm(self.samples, desc="Generating segments"):
            # 스켈레톤 파일 경로 찾기
            filename = sample['filename'].replace('.mp4', '_skeleton.json')
            skeleton_path = os.path.join(self.skeleton_base_path, filename)
            
            if not os.path.exists(skeleton_path):
                print(f"[WARNING] 스켈레톤 파일 없음: {skeleton_path}")
                continue
            
            # 스켈레톤 데이터 로드 (T, V, 3)
            pose_data = self._load_skeleton(skeleton_path)
            if pose_data is None:
                continue
            
            T, V, C = pose_data.shape  # (frames, 18, 3)
            
            # 관절 서브셋 적용
            if self.joint_subset is not None:
                pose_data = pose_data[:, self.joint_subset, :]  # (T, subset_V, 3)
            
            # 시퀀스 세그먼트 생성
            for start_idx in range(0, T - self.seg_len + 1, self.seg_stride):
                segment = pose_data[start_idx:start_idx + self.seg_len, :, :]  # (seg_len, V, 3)
                
                # 중앙 프레임 인덱스 계산
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
        
        # filter_label에 따라 세그먼트 필터링 (프레임 레벨)
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
            
            print(f"[DATASET] 필터링 ({self.filter_label}): {len(self.segments)}개 세그먼트")
        
        print(f"[DATASET] 총 {len(self.segments)}개 세그먼트 생성")
        print(f"[DATASET] 정상 세그먼트: {sum([1 for l in self.segment_labels if l == 0])}개")
        print(f"[DATASET] 이상 세그먼트: {sum([1 for l in self.segment_labels if l == 1])}개")
    
    def __len__(self):
        if self.load_per_batch:
            return len(self.segment_metadata)
        else:
            return len(self.segments)
    
    def __getitem__(self, idx):
        """
        STG-NF 방식의 전처리 적용
        Returns: (C, T, V) 형식의 텐서
        """
        # 전처리 캐시 확인 (augmentation이 없을 때만)
        if self.preprocessed_cache is not None and idx in self.preprocessed_cache:
            segment_tensor, label_tensor = self.preprocessed_cache[idx]
            return segment_tensor, label_tensor
        
        if self.load_per_batch:
            # 배치별 로딩: 메타데이터로부터 세그먼트 로드
            meta = self.segment_metadata[idx]
            pose_data = self._load_skeleton(meta['skeleton_path'])
            
            if pose_data is None:
                # 로드 실패 시 더미 데이터 반환
                V = len(self.joint_subset) if self.joint_subset else 18
                segment = np.zeros((self.seg_len, V, 3), dtype=np.float32)
                label = meta['label']
            else:
                # 관절 서브셋 적용
                if self.joint_subset is not None:
                    pose_data = pose_data[:, self.joint_subset, :]
                
                # 세그먼트 추출
                start_idx = meta['start_idx']
                segment = pose_data[start_idx:start_idx + self.seg_len, :, :]
                label = meta['label']
        else:
            # 사전 생성 방식 (기존)
            segment = np.array(self.segments[idx])
            label = self.segment_labels[idx]
        
        # Augmentation 적용 (Training 시에만)
        if self.apply_augmentation and random.random() < 0.5:
            transform = random.choice(self.transform_list)
            # (T, V, 3) -> (3, T, V) for transformation
            segment_transposed = segment.transpose(2, 0, 1)
            segment_transformed = transform(segment_transposed)
            # (3, T, V) -> (T, V, 3)
            segment = segment_transformed.transpose(1, 2, 0)
        
        # Normalization 적용 (STG-NF 방식)
        if self.normalize and normalize_pose is not None:
            # (T, V, 3) -> (1, T, V, 3) for batch processing
            segment_batch = segment[np.newaxis, ...]
            # STG-NF normalize_pose 함수 적용
            segment_normalized = normalize_pose(
                segment_batch, 
                vid_res=self.vid_res,
                symm_range=False
            )
            # (1, T, V, 3) -> (T, V, 3)
            segment = segment_normalized.squeeze(0)
        
        # (T, V, 3) -> (C=3, T, V) for PyTorch Conv
        segment = segment.transpose(2, 0, 1)
        
        # Use all channels: x, y, confidence (C=3)
        # segment shape: (3, T, V)
        
        segment_tensor = torch.FloatTensor(segment)
        label_tensor = torch.LongTensor([label])[0]
        
        # 전처리 캐시에 저장 (augmentation이 없을 때만)
        if self.preprocessed_cache is not None:
            self.preprocessed_cache[idx] = (segment_tensor, label_tensor)
        
        return segment_tensor, label_tensor
    
    def get_metadata(self, idx):
        """세그먼트 메타데이터 반환"""
        return self.segment_metadata[idx]


# ============================================================================
# 데이터 로더 생성 함수
# ============================================================================
def create_dataloader(json_path, skeleton_base_path, seg_len, seg_stride, 
                      joint_subset=None, batch_size=64, shuffle=True, 
                      num_workers=4, normalize=True, apply_augmentation=False, 
                      vid_res=[1920, 1080]):
    """
    DataLoader 생성 헬퍼 함수
    
    Args:
        json_path: 데이터 split JSON 경로
        skeleton_base_path: 스켈레톤 데이터 경로
        seg_len: 시퀀스 길이
        seg_stride: 세그먼트 stride
        joint_subset: 관절 서브셋 (None=전체)
        batch_size: 배치 크기
        shuffle: 데이터 셔플 여부
        num_workers: 워커 수
        normalize: 정규화 여부
        apply_augmentation: 데이터 증강 여부
        vid_res: 비디오 해상도
    
    Returns:
        DataLoader 객체
    """
    dataset = ShopliftingDataset(
        json_path=json_path,
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
