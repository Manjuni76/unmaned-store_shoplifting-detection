import numpy as np
from typing import Dict, Any, List, Tuple
import copy
import random

class KeypointPipeline:
    """
    키포인트 전처리 파이프라인 (keypoint_pipeline_bbox.py)
    
    [수정] 기본 정규화 방식을 'bbox' (Bounding Box)로 변경했습니다.
    """

    # 좌우 반전 짝 (OpenPose 18번 기준)
    COCO18_SWAP_PAIRS = [
        (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)
    ]

    def __init__(self,
                 conf_threshold: float = 0.3,
                 max_gap: int = 8,
                 smoothing_window_size: int = 8,
                 sequence_window_size: int = 24,
                 stride: int = 4,
                 # [!!!] 기본값을 'bbox'로 변경
                 norm_method: str = "bbox", 
                 frame_width: int = 1920, # bbox 방식에서는 참고용으로만 사용될 수 있음
                 frame_height: int = 1080,# bbox 방식에서는 참고용으로만 사용될 수 있음
                 aug_expansion_prob: float = 0.5,
                 aug_flip_prob: float = 0.5,
                 aug_shear_prob: float = 0.5,
                 aug_shear_range: float = 0.1):
        """
        [수정] norm_method 기본값을 'bbox'로 변경.
        """
        self.conf_threshold = conf_threshold
        self.max_gap = max_gap
        self.smoothing_window_size = smoothing_window_size
        self.sequence_window_size = sequence_window_size
        self.stride = stride
        self.norm_method = norm_method # 인스턴스 생성 시 'frame'으로 지정 가능
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.aug_expansion_prob = aug_expansion_prob
        self.aug_flip_prob = aug_flip_prob
        self.aug_shear_prob = aug_shear_prob
        self.aug_shear_range = aug_shear_range
        self.mean_coord = 0.0
        self.std_coord = 1.0

    # --- 1. Public Methods ---
    def process_for_training(self, data: Dict) -> List[Dict]:
        preprocessed_data = self._preprocess_frames(data)
        sequences = self._sliding_window_sequence_split(preprocessed_data)
        augmented_sequences = self._apply_augmentations(sequences)
        if augmented_sequences:
            self.fit_standardizer(augmented_sequences)
            standardized_sequences = self.apply_standardization(augmented_sequences)
            return standardized_sequences
        else:
            return []

    def process_for_evaluation(self, data: Dict) -> List[Dict]:
        preprocessed_data = self._preprocess_frames(data)
        sequences = self._sliding_window_sequence_split(preprocessed_data)
        if sequences:
            standardized_sequences = self.apply_standardization(sequences)
            return standardized_sequences
        else:
            return []

    # --- 2. Standardization ---
    def fit_standardizer(self, sequences: List[Dict[str, Any]]):
        all_coordinates = []
        for seq_data in sequences:
            sequence = seq_data["sequence"]
            valid_kps = sequence[sequence[..., 2] > 0]
            if valid_kps.shape[0] > 0:
                all_coordinates.append(valid_kps[:, :2].flatten())
        if not all_coordinates:
            self.mean_coord = 0.0
            self.std_coord = 1.0
            return
        coordinates_array = np.concatenate(all_coordinates)
        self.mean_coord = np.mean(coordinates_array)
        self.std_coord = np.std(coordinates_array)
        if self.std_coord < 1e-6:
            self.std_coord = 1.0
            
    def apply_standardization(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.std_coord == 0:
             return sequences
        standardized_sequences = []
        for seq_data in sequences:
            new_seq_data = seq_data.copy()
            sequence = new_seq_data["sequence"].copy()
            valid_mask = sequence[..., 2] > 0
            sequence[..., 0] = np.where(valid_mask, (sequence[..., 0] - self.mean_coord) / self.std_coord, 0)
            sequence[..., 1] = np.where(valid_mask, (sequence[..., 1] - self.mean_coord) / self.std_coord, 0)
            new_seq_data["sequence"] = sequence
            standardized_sequences.append(new_seq_data)
        return standardized_sequences

    # --- 3. Augmentation ---
    def _apply_augmentations(self, sequences: List[Dict]) -> List[Dict]:
        augmented_sequences = list(sequences)
        for seq_data in sequences:
            if random.random() < self.aug_expansion_prob:
                augmented_seq = seq_data["sequence"].copy()
                if random.random() < self.aug_flip_prob:
                    augmented_seq = self._horizontal_flip(augmented_seq)
                if random.random() < self.aug_shear_prob:
                    augmented_seq = self._shear(augmented_seq)
                new_seq_data = seq_data.copy()
                new_seq_data["sequence"] = augmented_seq
                new_seq_data["person_id"] = f"{seq_data['person_id']}_aug"
                augmented_sequences.append(new_seq_data)
        return augmented_sequences

    def _horizontal_flip(self, sequence: np.ndarray) -> np.ndarray:
        flipped_sequence = sequence.copy()
        for l_idx, r_idx in self.COCO18_SWAP_PAIRS:
            temp_l = flipped_sequence[:, l_idx, :].copy()
            flipped_sequence[:, l_idx, :] = flipped_sequence[:, r_idx, :]
            flipped_sequence[:, r_idx, :] = temp_l
        valid_mask = flipped_sequence[..., 2] > 0
        x_coords = flipped_sequence[..., 0]
        # BBox 정규화도 [0, 1] 범위이므로 1.0 - x 사용
        flipped_x = 1.0 - x_coords
        flipped_sequence[..., 0] = np.where(valid_mask, flipped_x, 0)
        return flipped_sequence

    def _shear(self, sequence: np.ndarray) -> np.ndarray:
        sheared_sequence = sequence.copy()
        shear_factor = random.uniform(-self.aug_shear_range, self.aug_shear_range)
        valid_mask = sheared_sequence[..., 2] > 0
        x_coords = sheared_sequence[..., 0]
        y_coords = sheared_sequence[..., 1]
        # BBox 정규화된 [0, 1] 좌표 기준으로 shear 적용
        sheared_x = x_coords + (shear_factor * y_coords)
        sheared_sequence[..., 0] = np.where(valid_mask, sheared_x, 0)
        return sheared_sequence

    # --- 4. Private Helper Methods ---
    def _preprocess_frames(self, data: Dict) -> Dict:
        data_copy = copy.deepcopy(data)
        filtered_data = self._filter_low_confidence(data_copy)
        interpolated_data = self._linear_interpolation(filtered_data)
        smoothed_data = self._n_frame_smoothing(interpolated_data)
        coco18_data = self._convert_to_coco18(smoothed_data)
        # _normalize_keypoints 함수가 self.norm_method 값을 보고 처리
        normalized_data = self._normalize_keypoints(coco18_data)
        return normalized_data

    def _filter_low_confidence(self, data: Dict) -> Dict:
        filtered_data = {}
        for person_id, frames in data.items():
            filtered_data[person_id] = {}
            for frame_num, frame_data in frames.items():
                keypoints = frame_data.get("keypoints", [])
                filtered_keypoints = []
                for kp in keypoints:
                    # 데이터 타입을 리스트로 가정 (json 로드 직후)
                    x, y, confidence = kp[0], kp[1], kp[2]
                    if x > 0 and y > 0 and confidence > self.conf_threshold:
                        filtered_keypoints.append([x, y, confidence])
                    else:
                        filtered_keypoints.append([0, 0, 0])
                filtered_data[person_id][frame_num] = {"keypoints": filtered_keypoints}
        return filtered_data

    def _linear_interpolation(self, data: Dict) -> Dict:
        result = {}
        for person_id, frames in data.items():
            result[person_id] = {}
            sorted_frames = sorted(frames.keys(), key=int)
            if not sorted_frames: continue
            num_frames = len(sorted_frames)
            num_joints = len(frames[sorted_frames[0]]["keypoints"])
            all_kps = np.zeros((num_frames, num_joints, 3))
            frame_map = {int(f): i for i, f in enumerate(sorted_frames)}
            for f_str, f_idx in frame_map.items():
                all_kps[f_idx] = np.array(frames[str(f_str)]["keypoints"])

            for j in range(num_joints):
                joint_seq = all_kps[:, j, :]
                conf = joint_seq[:, 2]
                valid_indices = np.where(conf > 0)[0]
                if len(valid_indices) < 2: continue
                invalid_indices = np.where(conf == 0)[0]
                for i in invalid_indices:
                    prev_idx_search = valid_indices[valid_indices < i]
                    prev_idx = prev_idx_search[-1] if len(prev_idx_search) > 0 else None
                    next_idx_search = valid_indices[valid_indices > i]
                    next_idx = next_idx_search[0] if len(next_idx_search) > 0 else None
                    if prev_idx is not None and next_idx is not None:
                        frame_gap = next_idx - prev_idx
                        if 0 < frame_gap <= self.max_gap:
                            ratio = (i - prev_idx) / frame_gap
                            interp_xy = joint_seq[prev_idx, :2] + (joint_seq[next_idx, :2] - joint_seq[prev_idx, :2]) * ratio
                            all_kps[i, j, :2] = interp_xy
                            all_kps[i, j, 2] = 0.5
            for f_str, f_idx in frame_map.items():
                result[person_id][f_str] = {"keypoints": all_kps[f_idx].tolist()}
        return result

    def _n_frame_smoothing(self, data: Dict) -> Dict:
        result = {}
        for person_id, frames in data.items():
            result[person_id] = {}
            sorted_frames = sorted(frames.keys(), key=int)
            if not sorted_frames: continue
            num_frames = len(sorted_frames)
            num_joints = len(frames[sorted_frames[0]]["keypoints"])
            all_kps = np.zeros((num_frames, num_joints, 3))
            frame_map = {int(f): i for i, f in enumerate(sorted_frames)}
            for f_str, f_idx in frame_map.items():
                all_kps[f_idx] = np.array(frames[str(f_str)]["keypoints"])

            smoothed_kps = all_kps.copy()
            half_window = self.smoothing_window_size // 2
            for i in range(num_frames):
                start = max(0, i - half_window)
                end = min(num_frames, i + half_window + 1)
                window = all_kps[start:end]
                for j in range(num_joints):
                    joint_window = window[:, j, :]
                    valid_kps = joint_window[joint_window[:, 2] > 0]
                    if len(valid_kps) > 0:
                        avg_kp = np.mean(valid_kps, axis=0)
                        smoothed_kps[i, j] = avg_kp
                    # else: # 유효값 없으면 원본 유지 (이미 smoothed_kps에 복사됨)
                    #     smoothed_kps[i, j] = all_kps[i, j]
            for f_str, f_idx in frame_map.items():
                result[person_id][f_str] = {"keypoints": smoothed_kps[f_idx].tolist()}
        return result

    def _convert_to_coco18(self, data: Dict) -> Dict:
        result = {}
        opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        for person_id, frames in data.items():
            result[person_id] = {}
            for frame_num, frame_data in frames.items():
                keypoints_17 = frame_data["keypoints"]
                keypoints_17_np = np.array(keypoints_17)
                if keypoints_17_np.shape[0] != 17: # 이미 18개면 건너뛰기
                     result[person_id][frame_num] = frame_data
                     continue
                left_shoulder = keypoints_17_np[5]
                right_shoulder = keypoints_17_np[6]
                if left_shoulder[2] > 0 and right_shoulder[2] > 0:
                    neck_xy = (left_shoulder[:2] + right_shoulder[:2]) / 2
                    neck_conf = min(left_shoulder[2], right_shoulder[2])
                    neck = np.array([neck_xy[0], neck_xy[1], neck_conf])
                else:
                    neck = np.array([0, 0, 0])
                keypoints_18_temp = np.concatenate([keypoints_17_np, neck.reshape(1, 3)], axis=0)
                keypoints_18_final = keypoints_18_temp[opp_order, :]
                result[person_id][frame_num] = {"keypoints": keypoints_18_final.tolist()}
        return result

    def _normalize_keypoints(self, data: Dict) -> Dict:
        """ 4. 정규화 (self.norm_method에 따라 'bbox' 또는 'frame') """
        result = {}
        for person_id, frames in data.items():
            result[person_id] = {} # 각 사람별 결과를 저장할 딕셔너리

            # --- BBox 방식 ---
            if self.norm_method == "bbox":
                all_x, all_y = [], []
                # 해당 사람의 모든 프레임에서 유효 키포인트 좌표 수집
                for frame_data in frames.values():
                    # frame_data['keypoints']가 리스트 형태라고 가정
                    for kp in frame_data.get("keypoints", []):
                        if kp[2] > 0: # confidence > 0
                            all_x.append(kp[0])
                            all_y.append(kp[1])

                # 유효 키포인트가 있어야 BBox 계산 가능
                if all_x and all_y:
                    min_x, max_x = min(all_x), max(all_x)
                    min_y, max_y = min(all_y), max(all_y)
                    # 너비/높이가 0이 되는 것 방지
                    bbox_width = max(max_x - min_x, 1.0)
                    bbox_height = max(max_y - min_y, 1.0)

                    # 각 프레임별로 정규화 적용
                    for frame_num, frame_data in frames.items():
                        normalized_keypoints = []
                        for kp in frame_data.get("keypoints", []):
                            if kp[2] > 0:
                                norm_x = (kp[0] - min_x) / bbox_width
                                norm_y = (kp[1] - min_y) / bbox_height
                                normalized_keypoints.append([norm_x, norm_y, kp[2]])
                            else:
                                normalized_keypoints.append([0, 0, 0])
                        result[person_id][frame_num] = {"keypoints": normalized_keypoints}
                else:
                    # 유효 키포인트 없으면 원본 프레임 구조 유지 (모두 [0,0,0])
                    for frame_num, frame_data in frames.items():
                         normalized_keypoints = [[0,0,0]] * len(frame_data.get("keypoints",[]))
                         result[person_id][frame_num] = {"keypoints": normalized_keypoints}


            # --- Frame 방식 ---
            elif self.norm_method == "frame":
                for frame_num, frame_data in frames.items():
                    normalized_keypoints = []
                    for kp in frame_data.get("keypoints", []):
                        if kp[2] > 0:
                            norm_x = kp[0] / self.frame_width
                            norm_y = kp[1] / self.frame_height
                            normalized_keypoints.append([norm_x, norm_y, kp[2]])
                        else:
                            normalized_keypoints.append([0, 0, 0])
                    result[person_id][frame_num] = {"keypoints": normalized_keypoints}
            
            # --- 지원하지 않는 방식 ---
            else:
                 raise ValueError(f"지원하지 않는 정규화 방식입니다: {self.norm_method}")

        return result
        
    def _sliding_window_sequence_split(self, data: Dict) -> List[Dict[str, Any]]:
        sequences = []
        for person_id, frames in data.items():
            sorted_frame_nums = sorted(frames.keys(), key=int)
            if len(sorted_frame_nums) < self.sequence_window_size: continue
            
            num_frames = len(sorted_frame_nums)
            num_joints = len(frames[sorted_frame_nums[0]]["keypoints"]) # 18
            all_kps = np.zeros((num_frames, num_joints, 3))
            frame_map = {int(f): i for i, f in enumerate(sorted_frame_nums)}
            for f_str, f_idx in frame_map.items():
                all_kps[f_idx] = np.array(frames[str(f_str)]["keypoints"])

            for start_idx in range(0, num_frames - self.sequence_window_size + 1, self.stride):
                end_idx = start_idx + self.sequence_window_size
                sequence_array = all_kps[start_idx:end_idx]
                
                # 시퀀스 유효성 검사 (평균 신뢰도 기반)
                avg_conf = np.mean(sequence_array[..., 2][sequence_array[..., 2] > 0]) # 유효값만 평균
                if np.isnan(avg_conf) or avg_conf < 0.1: # 유효값 없거나 너무 낮으면 스킵
                    continue
                    
                window_frame_nums = sorted_frame_nums[start_idx:end_idx]
                sequences.append({
                    "person_id": person_id,
                    "sequence": sequence_array,
                    "frames": window_frame_nums,
                    "start_frame": int(window_frame_nums[0]),
                    "end_frame": int(window_frame_nums[-1])
                })
        return sequences
