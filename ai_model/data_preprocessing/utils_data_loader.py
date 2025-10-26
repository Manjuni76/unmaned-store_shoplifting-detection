import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List
import pickle

# --- 1. 'dataset.py'와 동일한 COCO18 관절 인덱스 정의 ---
# [ 0:코, 1:목, 2:R어깨, 3:R팔꿈치, 4:R손목, 5:L어깨, 6:L팔꿈치, 7:L손목,
#   8:R골반, 9:R무릎, 10:R발목, 11:L골반, 12:L무릎, 13:L발목,
#   14:R눈, 15:L눈, 16:R귀, 17:L귀 ]

# 학습시킬 부위와 해당 관절 인덱스 맵
JOINT_MAP = {
    'all': list(range(18)),
    'head': [0, 1, 14, 15, 16, 17],    # 6개
    'torso': [1, 2, 5, 8, 11],        # 5개 (논문 'body'와 동일)
    'arms': [2, 3, 4, 5, 6, 7],       # 6개
    'legs': [8, 9, 10, 11, 12, 13]  # 6개
}

# 'graph.py'의 'openpose' 레이아웃과 동일한 전신 뼈대
# (주의: dataset.py의 재정렬된 인덱스 기준이 *아니라*, graph.py의 원본 인덱스 기준)
ALL_BONES_GRAPH_PY = [
    (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
    (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)
]

# --- 2. 'graph.py'의 유틸리티 함수 (그대로 가져옴) ---
def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

# --- 3. 부위별 그래프(A) 생성 함수 ---
def get_graph_for_part(part_name: str, strategy='spatial', max_hop=1):
    """
    지정된 부위(part_name)에 대한 ST-GCN 인접 행렬(A)을 생성합니다.
    """
    if part_name not in JOINT_MAP:
        raise ValueError(f"'{part_name}'은 유효한 부위가 아닙니다.")

    # 이 부위에 해당하는 관절 인덱스 (논문 dataset.py 기준)
    joint_indices_data = JOINT_MAP[part_name]
    num_node = len(joint_indices_data)

    # 부위 내 로컬 인덱스 맵 (0 ~ num_node-1)
    # 예: 'arms' [2, 3, 4, 5, 6, 7] -> {2:0, 3:1, 4:2, 5:3, 6:4, 7:5}
    global_to_local_idx = {global_idx: local_idx for local_idx, global_idx in enumerate(joint_indices_data)}

    # 부위에 속하는 뼈 필터링 (로컬 인덱스 기준)
    part_bones = []
    for (i_global, j_global) in ALL_BONES_GRAPH_PY: # graph.py의 뼈대 사용
        if i_global in global_to_local_idx and j_global in global_to_local_idx:
            i_local = global_to_local_idx[i_global]
            j_local = global_to_local_idx[j_global]
            part_bones.append((i_local, j_local))

    self_link = [(i, i) for i in range(num_node)]
    edge = self_link + part_bones

    hop_dis = get_hop_distance(num_node, edge, max_hop=max_hop)

    adjacency = np.zeros((num_node, num_node))
    valid_hop = range(0, max_hop + 1)
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalize_adjacency = normalize_digraph(adjacency)

    if strategy == 'spatial':
        A = []
        # 부위의 중심 노드 결정 (논문처럼 목(1)이 있으면 사용, 없으면 첫번째 관절)
        center_global = 1 # 목
        if center_global in global_to_local_idx:
            center = global_to_local_idx[center_global]
        else:
            center = 0 # 부위의 첫 번째 관절 사용

        for hop in valid_hop:
            a_root = np.zeros((num_node, num_node))
            a_close = np.zeros((num_node, num_node))
            a_further = np.zeros((num_node, num_node))
            for i in range(num_node):
                for j in range(num_node):
                    if hop_dis[j, i] == hop:
                        # 중심과의 거리 비교
                        dist_j_center = hop_dis[j, center]
                        dist_i_center = hop_dis[i, center]

                        if dist_j_center == dist_i_center:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif dist_j_center > dist_i_center: # j가 중심에서 더 멀리 있음 -> i는 중심에 가까움 (close)
                             a_close[j, i] = normalize_adjacency[j, i]
                        else: # j가 중심에 더 가까이 있음 -> i는 중심에서 멂 (further)
                            a_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_root + a_close) # 수정: a_close가 아니라 a_root + a_close
                A.append(a_further)
        A = np.stack(A)
        return A # (K, V, V) 형태
    else:
        raise NotImplementedError("spatial 전략만 구현되었습니다.")


# --- 4. 부위별 PyTorch 데이터셋 클래스 ---
class BodyPartDataset(Dataset):
    """
    미리 전처리된 시퀀스 리스트(.pkl)를 읽어,
    지정된 부위(part_name)의 관절만 필터링하여 텐서로 반환합니다.
    """
    def __init__(self, pkl_file_path: str, part_name: str):
        self.part_name = part_name

        if part_name not in JOINT_MAP:
            raise ValueError(f"'{part_name}'은 유효한 부위가 아닙니다.")

        # 1. pkl 파일 로드
        print(f"[{part_name} Dataset] '{pkl_file_path}' 로드 중...")
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)
        self.all_sequences = data["sequences"] # 전처리된 시퀀스 리스트
        # (참고) self.global_mean = data["global_mean"]
        # (참고) self.global_std = data["global_std"]

        # 2. 이 부위에 해당하는 관절 인덱스 (논문 dataset.py 기준)
        self.joint_indices = JOINT_MAP[part_name]

        print(f"[{part_name} Dataset] 총 {len(self.all_sequences)}개 시퀀스 로드 완료.")
        print(f"[{part_name} Dataset] 관절 {len(self.joint_indices)}개 필터링: {self.joint_indices}")

    def __len__(self) -> int:
        return len(self.all_sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        # 1. 전처리/표준화가 완료된 시퀀스 딕셔너리
        seq_data = self.all_sequences[index]

        # 2. (T, 18, 3) 형태의 numpy 배열 (T=sequence_window_size)
        sequence_np = seq_data["sequence"]

        # 3. [핵심] 부위별 관절 필터링
        # (T, 18, 3) -> (T, V_part, 3)
        part_sequence_np = sequence_np[:, self.joint_indices, :]

        # 4. STG-NF 모델 입력 형태 (C, T, V)로 축 변환
        # (T, V_part, 3) -> (3, T, V_part)
        part_sequence_np = np.transpose(part_sequence_np, (2, 0, 1))

        # 5. 텐서로 변환 (C=3, T=24, V=num_part_joints)
        return torch.tensor(part_sequence_np, dtype=torch.float32)