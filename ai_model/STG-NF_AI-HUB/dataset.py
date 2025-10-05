import json
import math
import os
import re
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import normalize_pose
from utils.pose_utils import gen_clip_seg_data_np, get_ab_labels
from torch.utils.data import DataLoader

SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]

# COCO-18 관절 인덱스 정의
# 0: 코, 1: 목, 2: 오른쪽어깨, 3: 오른쪽팔꿈치, 4: 오른쪽손목
# 5: 왼쪽어깨, 6: 왼쪽팔꿈치, 7: 왼쪽손목, 8: 오른쪽골반, 9: 오른쪽무릎
# 10: 오른쪽발목, 11: 왼쪽골반, 12: 왼쪽무릎, 13: 왼쪽발목
# 14: 오른쪽눈, 15: 왼쪽눈, 16: 오른쪽귀, 17: 왼쪽귀

COCO18_ARMS = [2, 3, 4, 5, 6, 7]  # 전체 팔 관절
COCO18_LEGS = [8, 9, 10, 11, 12, 13]  # 전체 다리 관절
COCO18_BODY = [1, 2, 5, 8, 11]  # 몸통 관절 인덱스
COCO18_HEAD = [0, 14, 15, 16, 17]  # 머리 관절 인덱스

# 좌우 구분 관절 인덱스
COCO18_LEFT_ARM = [5, 6, 7]    # 왼쪽팔: 왼쪽어깨, 왼쪽팔꿈치, 왼쪽손목
COCO18_RIGHT_ARM = [2, 3, 4]   # 오른쪽팔: 오른쪽어깨, 오른쪽팔꿈치, 오른쪽손목
COCO18_LEFT_LEG = [11, 12, 13] # 왼쪽다리: 왼쪽골반, 왼쪽무릎, 왼쪽발목
COCO18_RIGHT_LEG = [8, 9, 10]  # 오른쪽다리: 오른쪽골반, 오른쪽무릎, 오른쪽발목
# Arm+body
COCO18_ARMS_BODY = [1,2,3,4,5,6,7,8,11]
COCO18_HEAD_BODY = [0,1,2,5,8,14,15,16,17]
# 관절 부위별 매핑 딕셔너리 정의 - train_eval.py와 동일하게 유지
JOINT_SUBSET_MAP = {
    'arms': COCO18_ARMS,                              # [2,3,4,5,6,7] 전체 팔
    'legs': COCO18_LEGS,                              # [8,9,10,11,12,13] 전체 다리
    'left_arm': COCO18_LEFT_ARM,                      # [5,6,7] 왼쪽팔만
    'right_arm': COCO18_RIGHT_ARM,                    # [2,3,4] 오른쪽팔만
    'left_leg': COCO18_LEFT_LEG,                      # [11,12,13] 왼쪽다리만
    'right_leg': COCO18_RIGHT_LEG,                    # [8,9,10] 오른쪽다리만
    'body': COCO18_BODY,                              # [1,2,5,8,11] 몸통
    'head': COCO18_HEAD,                              # [0,14,15,16,17] 머리
    'arm+body':COCO18_ARMS_BODY,# 상체 전체
    'head+body':COCO18_HEAD_BODY,
    'all': None                                       # None은 모든 관절을 의미
}

# Dataset 상속 -> DataLoader와 함께 쓰기 위해 만듦
class PoseSegDataset(Dataset):
    """
    Generates a dataset with two objects, a np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced seq


    If path_to_patches is provided uses pre-extracted patches. If lmdb_file or vid_dir are
    provided extracts patches from them, while hurting performance.
    """

    # 데이터셋 처음 불러올 때 데이터 준비 단계 전체 수행하는 역할
    # -> 경로, 데이터 변환(증강), 관절 선택, 데이터 실제 불러오기, 이상 데이터 추가, 메타데이터 저장
    def __init__(self, path_to_json_dir, path_to_vid_dir=None, normalize_pose_segs=True, return_indices=False,
                 return_metadata=False, debug=False, return_global=True, evaluate=False, abnormal_train_path=None,
                 **dataset_args):
        super().__init__()
        self.args = dataset_args
        self.path_to_json = path_to_json_dir
        self.patches_db = None
        self.use_patches = False
        self.normalize_pose_segs = normalize_pose_segs
        self.headless = dataset_args.get('headless', False)
        self.path_to_vid_dir = path_to_vid_dir
        self.eval = evaluate
        self.debug = debug
        num_clips = dataset_args.get('specific_clip', None)
        self.return_indices = return_indices
        self.return_metadata = return_metadata
        self.return_global = return_global
        self.transform_list = dataset_args.get('trans_list', None)
        
        # 관절 서브셋 처리
        self.joint_subset_name = dataset_args.get('joint_subset', None)
        self.joint_subset = JOINT_SUBSET_MAP.get(self.joint_subset_name, None)
        if self.transform_list is None:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(self.transform_list)
        self.train_seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        # joint_subset이 dataset_args에 이미 포함되어 있으므로 따로 전달하지 않음
        dataset_args_copy = dataset_args.copy()
        if 'joint_subset' in dataset_args_copy:
            del dataset_args_copy['joint_subset']  # dataset_args에서 제거
            
        self.segs_data_np, self.segs_meta, self.person_keys, self.global_data_np, \
        self.global_data, self.segs_score_np = \
            gen_dataset(path_to_json_dir, num_clips=num_clips, ret_keys=True,
                        ret_global_data=return_global, joint_subset=self.joint_subset, **dataset_args_copy)
        
        # 데이터셋 정보 출력
        if self.joint_subset is not None:
            print(f"[DATASET] 선택된 관절 부위: {self.joint_subset_name}, 관절 수: {len(self.joint_subset)}")
            print(f"[DATASET] 데이터 크기: {self.segs_data_np.shape}")
        self.segs_meta = np.array(self.segs_meta)
        
        # abnormal_train_path가 있으면 비정상 데이터도 로드
        if abnormal_train_path is not None:
            # 비정상 데이터에도 동일한 joint_subset 적용
            self.segs_data_np_ab, self.segs_meta_ab, self.person_keys_ab, self.global_data_np_ab, \
            self.global_data_ab, self.segs_score_np_ab = \
                gen_dataset(abnormal_train_path, num_clips=num_clips, ret_keys=True,
                            ret_global_data=return_global, joint_subset=self.joint_subset, **dataset_args_copy)
            self.segs_meta_ab = np.array(self.segs_meta_ab)
            ab_labels = get_ab_labels(self.segs_data_np_ab, self.segs_meta_ab, path_to_vid_dir, abnormal_train_path)
            num_normal_samp = self.segs_data_np.shape[0]
            num_abnormal_samp = (ab_labels == -1).sum()
            total_num_normal_samp = num_normal_samp + (ab_labels == 1).sum()
            print("Num of abnormal sapmles: {}  | Num of normal samples: {}  |  Precent: {}".format(
                num_abnormal_samp, total_num_normal_samp, num_abnormal_samp / total_num_normal_samp))
            self.labels = np.concatenate((np.ones(num_normal_samp), ab_labels),
                                         axis=0).astype(int)
            self.segs_data_np = np.concatenate((self.segs_data_np, self.segs_data_np_ab), axis=0)
            self.segs_meta = np.concatenate((self.segs_meta, self.segs_meta_ab), axis=0)
            self.global_data_np = np.concatenate((self.global_data_np, self.global_data_np_ab), axis=0)
            self.segs_score_np = np.concatenate(
                (self.segs_score_np, self.segs_score_np_ab), axis=0)
            self.global_data += self.global_data_ab
            self.person_keys.update(self.person_keys_ab)
        else:
            self.labels = np.ones(self.segs_data_np.shape[0])
        # Convert person keys to ints
        self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    # 데이터셋 샘플 1개 가져와 증강 여부를 확인 후 정규화 및 학습에 쓸 수 있는 텐서로 변환
    # 훈련할 때 얘를 1개씩 가져가서 정규화를 시키고 하는 거임 -> 메모리 이슈 방지를 위해 (효율성)
    # 추가로 매번 데이터를 랜덤하게 변형 하는 게 성능에 좋음
    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = math.floor(index / self.num_samples)
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
        else:
            sample_index = index
            data_transformed = np.array(self.segs_data_np[index])
            trans_index = 0  # No transformations

        if self.normalize_pose_segs:
            data_transformed = normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...],
                                              **self.args).squeeze(axis=0).transpose(2, 0, 1)

        ret_arr = [data_transformed, trans_index]
        ret_arr += [self.segs_score_np[sample_index]]
        ret_arr += [self.labels[sample_index]]
        return ret_arr

    # 데이터셋 전체를 numpy 형태로 꺼내고 싶을 때 사용, 주로 디버깅 / 분석용임
    # normalize_poze 적용 여부에 따라 전체 데이터 정규화
    def get_all_data(self, normalize_pose_segs=True):
        if normalize_pose_segs:
            segs_data_np = normalize_pose(self.segs_data_np.transpose((0, 2, 3, 1)), **self.args).transpose(
                (0, 3, 1, 2))
        else:
            segs_data_np = self.segs_data_np
        if self.num_transform == 1 or self.eval:
            return list(segs_data_np)
        return segs_data_np
    
    # len(dataset)할 때 불림 전체 데이터 크기 반환
    def __len__(self):
        return self.num_transform * self.num_samples

#  DataSet이랑 DataLoader을 생성하는 함수라고 보면 됨
    #batch_size: 한 번 학습할 때 몇 개의 샘플을 묶어서 가져올지 (mini-batch 크기).
    #num_workers: 데이터를 읽어오는 프로세스 수. (여러 개로 하면 병렬로 빨라짐).
    #pin_memory=True: GPU 학습 시 CPU → GPU 메모리 복사 속도를 빠르게 해줌.
def get_dataset_and_loader(args, trans_list, only_test=False):
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset_args = {
        'headless': args.headless,
        'scale': args.norm_scale,
        'scale_proportional': args.prop_norm_scale,
        'seg_len': args.seg_len,
        'return_indices': True,
        'return_metadata': True,
        "dataset": args.dataset,
        'train_seg_conf_th': args.train_seg_conf_th,
        'specific_clip': args.specific_clip,
        'joint_subset': getattr(args, 'joint_subset', None),   #main에서 고른 관절 선택 옵션이 여기로 적용되는 거임
    }
    
    # 실제 관절 인덱스 가져오기
    if dataset_args['joint_subset'] is not None:
        joint_name = dataset_args['joint_subset']
        print(f"[LOADER] 선택된 관절 부위: {joint_name}")
        print(f"[LOADER] 해당 관절 인덱스: {JOINT_SUBSET_MAP.get(joint_name, None)}")
    dataset, loader = dict(), dict()  # 데이터셋과 로더를 담을 딕셔너리 초기화
    splits = ['train', 'test'] if not only_test else ['test']

    # split 별 데이터로더 생성
    for split in splits:
        evaluate = split == 'test'
        abnormal_train_path = args.pose_path_train_abnormal if split == 'train' else None # 테스트 시
        normalize_pose_segs = args.global_pose_segs 
        dataset_args['trans_list'] = trans_list[:args.num_transform] if split == 'train' else None # train시에는 증강 데이터도 적용 | test는 x
        dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 1  # No strides for test set train 데이터 늘리기
        dataset_args['vid_path'] = args.vid_path[split] # 비디오 경로
        dataset[split] = PoseSegDataset(args.pose_path[split], path_to_vid_dir=args.vid_path[split],
                                        normalize_pose_segs=normalize_pose_segs,
                                        evaluate=evaluate,
                                        abnormal_train_path=abnormal_train_path,
                                        **dataset_args)
        #DataLoader의 구조
        #DataLoader(
    #     dataset, 
    #     batch_size=1, 
    #     shuffle=False, 
    #     sampler=None, 
    #     batch_sampler=None, 
    #     num_workers=0, 
    #     collate_fn=None, 
    #     pin_memory=False, 
    #     drop_last=False, 
    #     timeout=0, 
    #     worker_init_fn=None,
    #     ...
    # ) -> 여기서 loader_args가 나머지 값들을 불러오는 거임
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    if only_test:
        loader['train'] = None
    return dataset, loader


def shanghaitech_hr_skip(shanghaitech_hr, scene_id, clip_id):
    if not shanghaitech_hr:
        return shanghaitech_hr
    if (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
        return True
    return False


# Json파일을 읽어서 포즈시퀀스, 신뢰도, 메타데이터, 글로벌 포즈 데이터를 numpy 배열로 변환하고 정리하는 역할
def gen_dataset(person_json_root, num_clips=None, kp18_format=True, ret_keys=False, ret_global_data=True,
                joint_subset=None, **dataset_args):
    #arg에서 설정한 값들 불러오고 np로 변환할 리스트를 생성
    segs_data_np = []
    segs_score_np = []
    segs_meta = []
    global_data = []
    person_keys = dict()
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 24)
    headless = dataset_args.get('headless', False)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
    dataset = dataset_args.get('dataset', 'PoseLift')    

    dir_list = os.listdir(person_json_root)
    if dataset_args.get('dataset') == 'Shoplifting':
        json_list = sorted([fn for fn in dir_list if fn.endswith('.h5')])
    else:
        json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')]) #Json 파일 불러오기 사람볋로
    if num_clips is not None:
        json_list = [json_list[num_clips]]  # For debugging purposes

    # 데이터셋 구분 처리 (UBnormal이랑 다른 데이터셋)
    for person_dict_fn in tqdm(json_list):
        if dataset == "UBnormal":
            type, scene_id, clip_id = \
                re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*', person_dict_fn)[0]
            clip_id = type + "_" + clip_id
        elif dataset == "Shoplifting":
            # Shoplifting 파일명 예:
            #  - abnormal_shoplifting_001_C_3_12_16_...
            #  - normal_shopping_001_C_1_1_10_...
            # clip_id는 파일명 앞쪽의 3자리 숫자(001 등)로 사용, scene_id는 'C' 다음 첫 숫자를 사용
            parts = person_dict_fn.split('_')
            try:
                clip_id = int(parts[2])  # 001 -> 1
            except Exception:
                clip_id = 0
            try:
                # 'C' 다음의 첫 숫자 (예: ... _C_1_1_10_ ... -> scene_id=1)
                scene_id = int(parts[4])
            except Exception:
                scene_id = 0
        else:
            scene_id, clip_id = person_dict_fn.split('_')[:2]
            if shanghaitech_hr_skip(dataset=="ShaghaiTech-HR", scene_id, clip_id):
                continue
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        if dataset == "Shoplifting":
            import h5py
            with h5py.File(clip_json_path, 'r') as f:
                # HDF5 구조:
                #   <person_group_name> -> datasets: 'frame_numbers' (T,), 'keypoints' (T, J, 3)
                # 단, 다운스트림은 다음 형태를 기대:
                #   clip_dict[person_id(str)] [frame_num(str)] = {'keypoints': (J,3)list, 'scores': float}
                clip_dict = {}
                for person_group_name in f.keys():
                    # person id 숫자 추출 (예: ..._person_16 -> 16)
                    m = re.search(r'person_(\d+)$', person_group_name)
                    if m:
                        pid = m.group(1)
                    else:
                        # 숫자만 추출, 없으면 0
                        pid = re.sub(r'\D', '', person_group_name) or '0'

                    grp = f[person_group_name]
                    # keypoints 필수
                    if 'keypoints' not in grp:
                        # 지원 불가 구조는 스킵
                        continue
                    keypoints = grp['keypoints'][()]
                    # frame_numbers가 없을 수 있으므로 대응
                    if 'frame_numbers' in grp:
                        frame_numbers = grp['frame_numbers'][()]
                    elif 'frames' in grp:
                        frame_numbers = grp['frames'][()]
                    else:
                        frame_numbers = None

                    # per-person dict 구성
                    per_person = {}
                    # frame_numbers를 int 리스트로 변환
                    if frame_numbers is not None:
                        try:
                            frames_list = [int(x) for x in frame_numbers]
                        except Exception:
                            frames_list = list(range(len(keypoints)))
                    else:
                        frames_list = list(range(len(keypoints)))

                    for i, fn in enumerate(frames_list):
                        kp = keypoints[i]
                        kp_list = kp.tolist()
                        # score는 신뢰도 평균 사용 (가능하면), 실패 시 1.0
                        try:
                            import numpy as _np  # 로컬 임포트(안전)
                            score_val = float(_np.nanmean(_np.array(kp)[..., 2]))
                        except Exception:
                            score_val = 1.0
                        per_person[str(int(fn))] = {'keypoints': kp_list, 'scores': score_val}

                    if per_person:
                        clip_dict[str(int(pid))] = per_person
        else:
            with open(clip_json_path, 'r') as f:
                clip_dict = json.load(f)
        clip_segs_data_np, clip_segs_meta, clip_keys, single_pos_np, _, score_segs_data_np = gen_clip_seg_data_np(
            clip_dict, start_ofst,
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            ret_keys=ret_keys,
            dataset=dataset)

        _, _, _, global_data_np, global_data, _ = gen_clip_seg_data_np(clip_dict, start_ofst, 1, 1, scene_id=scene_id,
                                                                       clip_id=clip_id,
                                                                       ret_keys=ret_keys,
                                                                       global_pose_data=global_data,
                                                                       dataset=dataset)
        segs_data_np.append(clip_segs_data_np)
        segs_score_np.append(score_segs_data_np)
        segs_meta += clip_segs_meta
        person_keys = {**person_keys, **clip_keys}

    # Global data
    global_data_np = np.expand_dims(np.concatenate(global_data, axis=0), axis=1)
    segs_data_np = np.concatenate(segs_data_np, axis=0)
    segs_score_np = np.concatenate(segs_score_np, axis=0)

    # if normalize_pose_segs:
    #     segs_data_np = normalize_pose(segs_data_np, vid_res=vid_res, **dataset_args)
    #     global_data_np = normalize_pose(global_data_np, vid_res=vid_res, **dataset_args)
    #     global_data = [normalize_pose(np.expand_dims(data, axis=0), **dataset_args).squeeze() for data
    #                    in global_data]
    if kp18_format and segs_data_np.shape[-2] == 17:
        segs_data_np = keypoints17_to_coco18(segs_data_np)
        global_data_np = keypoints17_to_coco18(global_data_np)
        global_data = [keypoints17_to_coco18(data) for data in global_data]
    if headless:
        segs_data_np = segs_data_np[:, :, 5:]
        global_data_np = global_data_np[:, :, 5:]
        global_data = [data[:, 5:, :] for data in global_data]

    # 부위별 관절 인덱스 정의 제거 - 이미 상단에 JOINT_SUBSET_MAP으로 정의됨
    # 이 부분은 아래의 joint_subset 처리 코드와 통합됩니다.

    #시퀀스 데이터, 글로벌 데이터 shape 변환 (N, C, T, V) 이렇게 바꿔야 정규화 + 학습에 용이하다고 함
    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
    global_data_np = np.transpose(global_data_np, (0, 3, 1, 2)).astype(np.float32)

    # 관절 부위 필터링 적용 (모든 부위에 대해 일관되게 처리)
    if joint_subset is not None:
        print(f"[GEN_DATASET] 관절 부위 필터링 적용 전 shape: {segs_data_np.shape}")
        # 데이터 형태는 (N, C, T, V)이므로 마지막 차원을 필터링
        segs_data_np = segs_data_np[:, :, :, joint_subset]
        global_data_np = global_data_np[:, :, :, joint_subset] if global_data_np.shape[3] > len(joint_subset) else global_data_np
        # 개별 데이터 슬라이싱 (data 변수 참조 오류 수정)
        if len(global_data) > 0:
            try:
                global_data = [g_data[:, joint_subset, :] for g_data in global_data if g_data.shape[1] > len(joint_subset)]
            except Exception as e:
                print(f"[WARNING] global_data 슬라이싱 오류: {e}")
        
        # 관절별 점수도 있으면 함께 슬라이스
        try:
            segs_score_np = np.asarray(segs_score_np)
            if segs_score_np.ndim == 2 and joint_subset is not None and segs_score_np.shape[1] >= max(joint_subset) + 1:
                segs_score_np = segs_score_np[:, joint_subset]
        except Exception as e:
            print(f"[WARNING] 관절 점수 슬라이싱 오류: {e}")
        
        print(f"[GEN_DATASET] 관절 부위 필터링 적용 후 shape: {segs_data_np.shape}")
        # 관절별 confidence 값 디버깅
    print("[DEBUG] segs_score_np shape:", segs_score_np.shape)
    print("[DEBUG] segs_score_np min:", segs_score_np.min())
    print("[DEBUG] segs_score_np max:", segs_score_np.max())
    print("[DEBUG] segs_score_np mean:", segs_score_np.mean())
        
    if seg_conf_th > 0.0:
        segs_data_np, segs_meta, segs_score_np = \
            seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th)
    if ret_global_data:
        if ret_keys:
            return segs_data_np, segs_meta, person_keys, global_data_np, global_data, segs_score_np
        else:
            return segs_data_np, segs_meta, global_data_np, global_data, segs_score_np
    if ret_keys:
        return segs_data_np, segs_meta, person_keys, segs_score_np
    else:
        return segs_data_np, segs_meta, segs_score_np

#목 추가하고 COCO18로 인덱스 변환한 거
def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    #run_experiment사용할때
    kp_np = np.array(kps)
    #kp_np = np.array(kps, dtype=np.float32)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=int)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18.astype(np.float32)

# 신뢰도 너무 낮은구간 걸러냄
def seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th=2.0):
    # seg_len = segs_data_np.shape[2]
    # conf_vals = segs_data_np[:, 2]
    # sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
    sum_confs = segs_score_np.mean(axis=1)
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    segs_score_np = segs_score_np[sum_confs > seg_conf_th]

    return seg_data_filt, seg_meta_filt, segs_score_np


