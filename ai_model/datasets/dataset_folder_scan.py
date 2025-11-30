"""
JSON 없이 폴더 스캔 방식 Dataset (RAM Pre-loading 최적화 버전)
전처리 로직은 기존과 동일, Attention 훈련 속도을 높이기 위해 사용
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random

from .dataset import (
    interpolate_skeleton, 
    transform_identity, 
    transform_flip, 
    transform_shear, 
    transform_flip_shear
)
from args import Config

class FolderScanDataset(Dataset):
    """
    data 폴더를 직접 스캔하여 데이터 로드
    모든 데이터를 __init__ 시점에 전처리하여 RAM에 적재 (GPU 병목 해결) -> 데이터 용량이 크지 않아 가능
    """
    def __init__(self, skeleton_dir, gt_dir, seg_len=24, seg_stride=6,
                 joint_subset=None, normalize=True, apply_augmentation=False,
                 vid_res=[1920, 1080], use_cache=True, preprocess_cache=True):
        
        self.skeleton_dir = skeleton_dir
        self.gt_dir = gt_dir
        self.seg_len = seg_len
        self.seg_stride = seg_stride
        self.joint_subset = joint_subset
        self.normalize = normalize
        self.apply_augmentation = apply_augmentation
        self.vid_res = vid_res
        self.use_cache = use_cache
        self.preprocess_cache = preprocess_cache
        
        # 캐시
        self.skeleton_cache = {} if use_cache else None
        
        # 증강 변환 리스트
        if self.apply_augmentation:
            self.transform_list = [transform_identity, transform_flip, transform_shear, transform_flip_shear]
        else:
            self.transform_list = [transform_identity]
        
        # 1. 파일 스캔 & GT 매핑
        print(f"스켈레톤 폴더 스캔: {skeleton_dir}")
        skeleton_files = glob.glob(os.path.join(skeleton_dir, '*.json'))
        print(f"발견된 파일 수: {len(skeleton_files)}개")
        
        self.gt_map = {}
        if gt_dir and os.path.exists(gt_dir):
            for npy_file in glob.glob(os.path.join(gt_dir, '*.npy')):
                basename = os.path.basename(npy_file).replace('.npy', '')
                self.gt_map[basename] = npy_file
        
        # 2. 세그먼트 생성 (Raw Data)
        self.segments = []
        self.segment_labels = []
        self.segment_metadata = []
        
        self._generate_segments(skeleton_files)
        print(f"총 {len(self.segments)}개 세그먼트 생성됨")

        # [최적화] __getitem__에 있던 로직을 여기로 이동 (RAM 적재)
        print(f"모든 데이터를 RAM에 미리 전처리 중")
        self.data_buffer = []
        
        # 전처리용 상수 미리 계산
        vid_res_wconf = np.array(self.vid_res + [1], dtype=np.float32)

        for i in tqdm(range(len(self.segments)), desc="Pre-loading"):
            # Raw Segment 가져오기
            segment = np.array(self.segments[i])  # (T, V, 3)
            label = self.segment_labels[i]

            # --- [기존 전처리 로직 그대로 적용] ---
            
            # 1. 픽셀 정규화
            segment = segment / vid_res_wconf
            if Config.Data.SYMM_RANGE:
                segment[..., :2] = 2 * segment[..., :2] - 1
            
            # 2. 보간 (가장 느린 연산 -> 미리 수행)
            segment = interpolate_skeleton(segment)
            
            # 3. Transpose (T, V, 3) -> (3, T, V)
            segment_ctv = segment.transpose(2, 0, 1)
            
            # 4. 부위 선택
            if self.joint_subset is not None:
                segment_ctv = segment_ctv[:, :, self.joint_subset]
            
            # 5. x, y만 사용
            segment_xy = segment_ctv[:2, :, :]
            
            # 6. 중심화
            segment_mean = segment_xy.mean(axis=2, keepdims=True)
            segment_xy = segment_xy - segment_mean
            
            # 7. 스케일 정규화
            std_val = segment_xy.std()
            if std_val > 1e-6:
                segment_xy = segment_xy / std_val
            
            # 8. Tensor 변환 및 저장
            segment_tensor = torch.FloatTensor(segment_xy)
            label_tensor = torch.LongTensor([label])[0]
            
            self.data_buffer.append((segment_tensor, label_tensor))
            
        print(f"데이터 적재 완료! 학습 속도가 빨라집니다.")
        
        # 메모리 확보를 위해 Raw Data 삭제
        del self.segments
        self.skeleton_cache = None 

    def __len__(self):
        return len(self.data_buffer)
    
    def __getitem__(self, idx):
        """
        [최적화] 이미 전처리된 데이터를 꺼내기만 함 (0초 소요)
        증강(Augmentation)만 여기서 수행 (랜덤성 유지 위해)
        """
        # 1. RAM에서 꺼내기
        segment_tensor, label_tensor = self.data_buffer[idx]
        
        # 2. 증강 (학습 시 랜덤 적용)
        if self.apply_augmentation and random.random() < 0.5:
            # Tensor -> Numpy 변환 후 증강 적용
            segment_np = segment_tensor.numpy()
            transform = random.choice(self.transform_list)
            segment_np = transform(segment_np)
            # 다시 Tensor로
            segment_tensor = torch.FloatTensor(segment_np)
        
        return segment_tensor, label_tensor
    
    def get_metadata(self, idx):
        """메타데이터 반환"""
        return self.segment_metadata[idx]
    
    # 아래 헬퍼 함수들은 기존 코드 그대로 유지
    def _load_skeleton(self, skeleton_path):
        if self.use_cache and self.skeleton_cache is not None and skeleton_path in self.skeleton_cache:
            return self.skeleton_cache[skeleton_path]
        try:
            import json
            with open(skeleton_path, 'r', encoding='utf-8') as f:
                skeleton_data = json.load(f)
            if 'person_1' not in skeleton_data: return None
            person_data = skeleton_data['person_1']
            frame_ids = sorted([int(fid) for fid in person_data.keys()])
            frames = []
            for frame_id in frame_ids:
                frame_str = str(frame_id)
                if frame_str in person_data:
                    frame_info = person_data[frame_str]
                    if 'keypoints' in frame_info:
                        keypoints = np.array(frame_info['keypoints'], dtype=np.float32)
                        frames.append(keypoints)
            if len(frames) == 0: return None
            pose_data = np.stack(frames, axis=0)
            if pose_data.shape[1] == 17:
                pose_data = self._keypoints17_to_coco18(pose_data)
            filename = os.path.basename(skeleton_path)
            if filename.startswith('C_1_'):
                pose_data = pose_data[::3]
            if self.use_cache and self.skeleton_cache is not None:
                self.skeleton_cache[skeleton_path] = pose_data
            return pose_data
        except Exception as e:
            return None
    
    def _keypoints17_to_coco18(self, kps):
        kp_np = np.array(kps, dtype=np.float32)
        neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
        kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
        opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        opp_order = np.array(opp_order, dtype=int)
        kp_coco18 = kp_np[..., opp_order, :]
        return kp_coco18.astype(np.float32)
    
    def _generate_segments(self, skeleton_files):
        for skeleton_path in tqdm(skeleton_files, desc="Generating segments"):
            basename = os.path.basename(skeleton_path).replace('.json', '')
            pose_data = self._load_skeleton(skeleton_path)
            if pose_data is None: continue
            
            T, V, C = pose_data.shape
            gt_array = None
            if basename in self.gt_map:
                gt_array = np.load(self.gt_map[basename])
            
            for start_idx in range(0, T - self.seg_len + 1, self.seg_stride):
                segment = pose_data[start_idx:start_idx + self.seg_len, :, :]
                frame_sums = np.abs(segment).sum(axis=(1, 2))
                if (frame_sums > 1e-5).sum() < (self.seg_len // 2): continue
                
                center_frame_idx = start_idx + self.seg_len // 2
                if gt_array is not None and center_frame_idx < len(gt_array):
                    label = int(gt_array[center_frame_idx])
                else:
                    label = 0
                
                self.segments.append(segment)
                self.segment_labels.append(label)
                self.segment_metadata.append({
                    'filename': basename,
                    'start_frame': start_idx,
                    'center_frame': center_frame_idx
                })