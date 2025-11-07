"""
통합 학습 파이프라인:
1. STG-NF로 정상 데이터만 부위별 학습
2. STG-NF를 freeze
3. MLP 레이어를 추가하고 정상+이상 데이터로 학습
4. Test 데이터로 평가
"""

import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

# STG-NF_AI-HUB 경로 추가
stg_nf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'STG-NF_AI-HUB'))
if stg_nf_path not in sys.path:
    sys.path.insert(0, stg_nf_path)

# 관절 부위별 매핑 딕셔너리 (COCO-18 기준)
COCO18_ARMS = [2, 3, 4, 5, 6, 7]
COCO18_LEGS = [8, 9, 10, 11, 12, 13]
COCO18_BODY = [1, 2, 5, 8, 11]
COCO18_HEAD = [0, 14, 15, 16, 17]
COCO18_LEFT_ARM = [5, 6, 7]
COCO18_RIGHT_ARM = [2, 3, 4]
COCO18_LEFT_LEG = [11, 12, 13]
COCO18_RIGHT_LEG = [8, 9, 10]
COCO18_ARMS_BODY = [1, 2, 3, 4, 5, 6, 7, 8, 11]
COCO18_HEAD_BODY = [0, 1, 2, 5, 8, 14, 15, 16, 17]

JOINT_SUBSET_MAP = {
    'arms': COCO18_ARMS,
    'legs': COCO18_LEGS,
    'left_arm': COCO18_LEFT_ARM,
    'right_arm': COCO18_RIGHT_ARM,
    'left_leg': COCO18_LEFT_LEG,
    'right_leg': COCO18_RIGHT_LEG,
    'body': COCO18_BODY,
    'head': COCO18_HEAD,
    'arm+body': COCO18_ARMS_BODY,
    'head+body': COCO18_HEAD_BODY,
    'all': None
}

# STG-NF 모델 import는 나중에 필요할 때
try:
    from models.STG_NF.model_pose import STG_NF  # type: ignore
    print("[SUCCESS] STG-NF 모델 import 성공")
except ImportError as e:
    print(f"[WARNING] STG-NF 모델 import 실패: {e}")
    print(f"[INFO] 런타임에 다시 시도합니다.")
    STG_NF = None  # type: ignore


def set_seed(seed=42):
    """모든 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[SEED] 모든 랜덤 시드를 {seed}로 고정했습니다.")


class ShopliftingDataset(Dataset):
    """
    data_split의 JSON 파일을 읽어서 스켈레톤 데이터를 로드하는 Dataset
    """
    def __init__(self, json_path, skeleton_base_path, seg_len=12, seg_stride=6, 
                 joint_subset=None, normalize=True):
        """
        Args:
            json_path: train_data.json, mlp_train_data.json, test_data.json 경로
            skeleton_base_path: 스켈레톤 데이터가 있는 기본 경로
            seg_len: 시퀀스 길이
            seg_stride: 시퀀스 stride
            joint_subset: 사용할 관절 인덱스 (None이면 전체)
            normalize: pose normalization 여부
        """
        self.seg_len = seg_len
        self.seg_stride = seg_stride
        self.joint_subset = joint_subset
        self.normalize = normalize
        self.skeleton_base_path = skeleton_base_path
        
        # JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 정상/이상 데이터 통합
        self.samples = []
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
        
        self._generate_segments()
    
    def _load_skeleton(self, skeleton_path):
        """스켈레톤 JSON 파일 로드"""
        try:
            with open(skeleton_path, 'r', encoding='utf-8') as f:
                skeleton_data = json.load(f)
            
            # 스켈레톤 데이터를 numpy 배열로 변환
            # 형식: (frames, joints, coords) -> (C=2, T=frames, V=joints)
            frames = []
            for frame_data in skeleton_data:
                if 'keypoints' in frame_data:
                    keypoints = np.array(frame_data['keypoints']).reshape(-1, 3)  # (V, 3)
                    xy = keypoints[:, :2]  # (V, 2) x,y만 사용
                    frames.append(xy)
            
            if len(frames) == 0:
                return None
            
            pose_data = np.stack(frames, axis=0)  # (T, V, 2)
            pose_data = pose_data.transpose(2, 0, 1)  # (2, T, V)
            
            return pose_data
        except Exception as e:
            print(f"[ERROR] 스켈레톤 로드 실패 {skeleton_path}: {e}")
            return None
    
    def _generate_segments(self):
        """영상별로 시퀀스 세그먼트 생성"""
        for sample in tqdm(self.samples, desc="Generating segments"):
            # 스켈레톤 파일 경로 찾기
            filename = sample['filename'].replace('.mp4', '_skeleton.json')
            skeleton_path = os.path.join(self.skeleton_base_path, filename)
            
            if not os.path.exists(skeleton_path):
                print(f"[WARNING] 스켈레톤 파일 없음: {skeleton_path}")
                continue
            
            # 스켈레톤 데이터 로드
            pose_data = self._load_skeleton(skeleton_path)
            if pose_data is None:
                continue
            
            C, T, V = pose_data.shape  # (2, frames, 18)
            
            # 관절 서브셋 적용
            if self.joint_subset is not None:
                pose_data = pose_data[:, :, self.joint_subset]  # (2, T, subset_V)
            
            # Normalization (선택)
            if self.normalize:
                pose_data = self._normalize_pose(pose_data)
            
            # 시퀀스 세그먼트 생성
            for start_idx in range(0, T - self.seg_len + 1, self.seg_stride):
                segment = pose_data[:, start_idx:start_idx + self.seg_len, :]
                
                # Ground truth 라벨 (이상 구간인지 확인)
                if sample['label'] == 1:  # 이상 데이터
                    theft_start = sample.get('theft_start', 0)
                    theft_end = sample.get('theft_end', T)
                    # 세그먼트가 이상 구간과 겹치면 1
                    seg_start = start_idx
                    seg_end = start_idx + self.seg_len - 1
                    if seg_end >= theft_start and seg_start <= theft_end:
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
                    'video_label': sample['label']
                })
        
        print(f"[DATASET] 총 {len(self.segments)}개 세그먼트 생성")
        print(f"[DATASET] 정상 세그먼트: {sum([1 for l in self.segment_labels if l == 0])}개")
        print(f"[DATASET] 이상 세그먼트: {sum([1 for l in self.segment_labels if l == 1])}개")
    
    def _normalize_pose(self, pose):
        """Pose normalization"""
        # 간단한 정규화: 평균을 빼고 표준편차로 나눔
        C, T, V = pose.shape
        pose_flat = pose.reshape(C, -1)
        mean = pose_flat.mean(axis=1, keepdims=True)
        std = pose_flat.std(axis=1, keepdims=True) + 1e-9
        pose_norm = (pose_flat - mean) / std
        return pose_norm.reshape(C, T, V)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        label = torch.LongTensor([self.segment_labels[idx]])[0]
        return segment, label


class STG_NF_with_MLP(nn.Module):
    """
    STG-NF + MLP 분류기
    STG-NF는 freeze하고 MLP만 학습
    """
    def __init__(self, stg_nf_model, feature_dim=256, num_classes=2):
        super().__init__()
        self.stg_nf = stg_nf_model
        
        # STG-NF freeze
        for param in self.stg_nf.parameters():
            param.requires_grad = False
        
        # MLP 분류기
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, V) pose sequence
        Returns:
            logits: (B, num_classes)
        """
        # STG-NF로 feature 추출
        with torch.no_grad():
            # STG-NF의 인코더 부분만 사용
            z, log_det = self.stg_nf.encode(x)
            features = z.view(z.size(0), -1)  # (B, feature_dim)
        
        # MLP 분류
        logits = self.mlp(features)
        return logits


def train_stg_nf(args):
    """
    Step 1: STG-NF로 정상 데이터만 학습
    """
    print("\n" + "="*80)
    print("STEP 1: STG-NF 정상 데이터 학습")
    print("="*80)
    
    # STG_NF 모델 런타임에 import
    global STG_NF
    if STG_NF is None:
        try:
            from models.STG_NF.model_pose import STG_NF  # type: ignore
            print("[SUCCESS] STG-NF 모델 런타임 import 성공")
        except ImportError as e:
            raise ImportError(f"STG-NF 모델을 import할 수 없습니다: {e}")
    
    # 데이터셋 로드 (train_data.json - 정상 데이터만)
    train_dataset = ShopliftingDataset(
        json_path=args['train_json'],
        skeleton_base_path=args['train_skeleton_path'],  # 수정
        seg_len=args['seg_len'],
        seg_stride=args['seg_stride'],
        joint_subset=args['joint_subset'],
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # STG-NF 모델 생성
    model_args = {
        'input_size': (2, args['seg_len'], len(args['joint_subset']) if args['joint_subset'] else 18),
        'K': 3,
        'L': 4,
        'hidden_channels': 512,
        'device': args['device'],
        'subset_idx': args['joint_subset']
    }
    
    model = STG_NF(**model_args).to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr_stgnf'])
    
    # 학습
    model.train()
    for epoch in range(args['epochs_stgnf']):
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data = data.to(args['device'])
            
            # 정상 데이터만 선택
            normal_mask = (labels == 0)
            if normal_mask.sum() == 0:
                continue
            data = data[normal_mask]
            
            optimizer.zero_grad()
            
            # STG-NF forward
            z, log_det = model(data)
            loss = model.loss(z, log_det)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args['epochs_stgnf']}, Loss: {avg_loss:.4f}")
    
    # 모델 저장
    torch.save(model.state_dict(), args['stgnf_checkpoint'])
    print(f"[SAVE] STG-NF 모델 저장: {args['stgnf_checkpoint']}")
    
    return model


def train_mlp_classifier(stg_nf_model, args):
    """
    Step 2: STG-NF freeze + MLP 학습 (정상+이상 데이터)
    """
    print("\n" + "="*80)
    print("STEP 2: MLP 분류기 학습 (정상+이상 데이터)")
    print("="*80)
    
    # 데이터셋 로드 (mlp_train_data.json - 정상+이상)
    mlp_train_dataset = ShopliftingDataset(
        json_path=args['mlp_train_json'],
        skeleton_base_path=args['mlp_skeleton_path'],  # 수정
        seg_len=args['seg_len'],
        seg_stride=args['seg_stride'],
        joint_subset=args['joint_subset'],
        normalize=True
    )
    
    mlp_train_loader = DataLoader(
        mlp_train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # STG-NF + MLP 모델
    full_model = STG_NF_with_MLP(
        stg_nf_model=stg_nf_model,
        feature_dim=256,
        num_classes=2
    ).to(args['device'])
    
    optimizer = torch.optim.Adam(full_model.mlp.parameters(), lr=args['lr_mlp'])
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    full_model.train()
    for epoch in range(args['epochs_mlp']):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(mlp_train_loader, desc=f"Epoch {epoch+1}")):
            data = data.to(args['device'])
            labels = labels.to(args['device'])
            
            optimizer.zero_grad()
            
            logits = full_model(data)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(mlp_train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{args['epochs_mlp']}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    # 모델 저장
    torch.save(full_model.state_dict(), args['full_model_checkpoint'])
    print(f"[SAVE] Full 모델 저장: {args['full_model_checkpoint']}")
    
    return full_model


def evaluate(model, args):
    """
    Step 3: Test 데이터로 평가
    """
    print("\n" + "="*80)
    print("STEP 3: Test 데이터 평가")
    print("="*80)
    
    # 테스트 데이터셋 로드
    test_dataset = ShopliftingDataset(
        json_path=args['test_json'],
        skeleton_base_path=args['test_skeleton_path'],  # 수정
        seg_len=args['seg_len'],
        seg_stride=args['seg_stride'],
        joint_subset=args['joint_subset'],
        normalize=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Evaluating"):
            data = data.to(args['device'])
            
            logits = model(data)
            scores = torch.softmax(logits, dim=1)[:, 1]  # 이상 클래스 확률
            _, preds = logits.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    # 메트릭 계산
    accuracy = (all_preds == all_labels).mean() * 100
    auc_roc = roc_auc_score(all_labels, all_scores)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    auc_pr = auc(recall, precision)
    
    print(f"\n[RESULTS]")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    
    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }


def main():
    # 베이스 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 하이퍼파라미터 설정
    args = {
        # 경로 설정
        'train_json': os.path.join(base_dir, 'data_split', 'output', 'train_data.json'),
        'mlp_train_json': os.path.join(base_dir, 'data_split', 'output', 'mlp_train_data.json'),
        'test_json': os.path.join(base_dir, 'data_split', 'output', 'test_data.json'),
        'train_skeleton_path': os.path.join(base_dir, 'data', 'train_data_skeleton_data'),
        'mlp_skeleton_path': os.path.join(base_dir, 'data', 'mlp_train_data_skeleton_data'),
        'test_skeleton_path': os.path.join(base_dir, 'data', 'test_data_skeleton_data'),
        
        # 체크포인트
        'stgnf_checkpoint': os.path.join(base_dir, 'ai_model', 'checkpoints', 'stgnf_arms.pth'),
        'full_model_checkpoint': os.path.join(base_dir, 'ai_model', 'checkpoints', 'full_model_arms.pth'),
        
        # 모델 설정
        'seg_len': 12,
        'seg_stride': 6,
        'joint_subset': JOINT_SUBSET_MAP['arms'],  # 'arms', 'legs', 'all' 등 선택
        
        # 학습 설정
        'batch_size': 32,
        'epochs_stgnf': 50,
        'epochs_mlp': 30,
        'lr_stgnf': 1e-4,
        'lr_mlp': 1e-3,
        
        # 기타
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }
    
    # 시드 설정
    set_seed(args['seed'])
    
    # Step 1: STG-NF 학습 (정상 데이터만)
    stg_nf_model = train_stg_nf(args)
    
    # Step 2: MLP 학습 (정상+이상 데이터)
    full_model = train_mlp_classifier(stg_nf_model, args)
    
    # Step 3: 평가
    results = evaluate(full_model, args)
    
    print("\n" + "="*80)
    print("학습 완료!")
    print("="*80)


if __name__ == '__main__':
    main()
