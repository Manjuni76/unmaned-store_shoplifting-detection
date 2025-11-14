"""
학습 설정 및 하이퍼파라미터 관리
모든 경로, 하이퍼파라미터, 모델 설정을 중앙에서 관리
"""

import os
import argparse


# ============================================================================
# 경로 설정
# ============================================================================
class PathConfig:
    """파일 경로 설정"""
    # 기본 경로
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    
    # 데이터 경로
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train_data_skeleton_data')
    MLP_TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'mlp_train_data_skeleton_data')
    TEST_DATA_DIR = os.path.join(DATA_DIR, 'test_data_skeleton_data')
    
    # Ground Truth 경로
    GT_DIR = os.path.join(DATA_DIR, 'gt')
    MLP_TRAIN_GT_DIR = os.path.join(GT_DIR, 'mlp_train_gt')
    TEST_GT_DIR = os.path.join(GT_DIR, 'test_gt')
    
    # 데이터 split JSON 경로
    DATA_SPLIT_DIR = os.path.join(PROJECT_ROOT, 'data_split', 'output')
    TRAIN_JSON = os.path.join(DATA_SPLIT_DIR, 'train_data.json')
    MLP_TRAIN_JSON = os.path.join(DATA_SPLIT_DIR, 'mlp_train_data.json')
    TEST_JSON = os.path.join(DATA_SPLIT_DIR, 'test_data.json')
    
    # 체크포인트 경로
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    
    # 결과 저장 경로
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    @classmethod
    def create_dirs(cls):
        """필요한 디렉토리 생성"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)


# ============================================================================
# 데이터 관련 설정
# ============================================================================
class DataConfig:
    """데이터 관련 설정"""
    # 비디오 해상도
    VID_RES = [1920, 1080]
    
    # 시퀀스 설정
    SEG_LEN = 12  # 시퀀스 길이 (프레임 수)
    
    # Stride 설정
    TRAIN_STRIDE = 3  # 학습 시 stride (겹침 허용)
    MLP_TRAIN_STRIDE = 2  # MLP 학습 시 stride
    EVAL_STRIDE = 1  # 평가 시 stride (모든 프레임 평가)
    
    # 정규화 설정
    NORMALIZE = True
    SYMM_RANGE = False  # [-1, 1] 범위로 매핑 여부
    
    # 데이터 증강 설정
    APPLY_AUGMENTATION = True  # 학습 시 증강 사용 여부
    
    # 데이터 캐싱 설정
    USE_CACHE = True  # 데이터를 메모리에 캐싱 (학습 속도 향상)
    CACHE_ALL_DATA = False  # 전체 데이터 캐싱 (메모리 충분할 때만)


# ============================================================================
# 관절 부위 매핑 (COCO-18 기준)
# ============================================================================
class JointConfig:
    """관절 부위 매핑"""
    # COCO-18 관절 인덱스
    COCO18_HEAD = [0, 14, 15, 16, 17]  # Nose, Eyes, Ears
    COCO18_ARMS = [2, 3, 4, 5, 6, 7]   # Shoulders, Elbows, Wrists
    COCO18_BODY = [1, 2, 5, 8, 11]     # Neck, Shoulders, Hips
    COCO18_LEGS = [8, 9, 10, 11, 12, 13]  # Hips, Knees, Ankles
    
    # 부위별 매핑
    JOINT_SUBSET_MAP = {
        'head': COCO18_HEAD,
        'arms': COCO18_ARMS,
        'body': COCO18_BODY,
        'legs': COCO18_LEGS,
        'all': None  # 전체 관절
    }
    
    # 학습할 부위 리스트
    BODY_PARTS = ['head', 'arms', 'body', 'legs', 'all']


# ============================================================================
# STG-NF 모델 하이퍼파라미터
# ============================================================================
class STGNFConfig:
    """STG-NF 모델 설정"""
    # 모델 아키텍처
    IN_CHANNELS = 3  # x, y, confidence
    HIDDEN_CHANNELS = 128  # Hidden channels for flow network
    HIDDEN_DIM = 64  # 레거시 호환용
    NUM_LAYERS = 8
    
    # Flow 모델 설정
    K = 8  # Number of flow steps
    L = 3  # Number of levels
    R = 2  # Number of repeats per level
    
    # Flow 설정
    ACTNORM_SCALE = 1.0
    FLOW_PERMUTATION = 'invconv'  # 'invconv', 'shuffle', 'reverse'
    FLOW_COUPLING = 'affine'  # 'additive', 'affine'
    LU_DECOMPOSED = True
    
    # 그래프 설정
    GRAPH_CFG = {
        'layout': 'openpose',
        'strategy': 'spatial',
        'max_hop': 1
    }
    EDGE_IMPORTANCE = False
    TEMPORAL_KERNEL_SIZE = 9
    STRATEGY = 'spatial'
    MAX_HOPS = 1
    LEARN_TOP = False
    
    # 학습 설정
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 256
    EPOCHS = 8
    
    # 최적화 설정
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 5.0  # Gradient clipping
    
    # 조기 종료 설정
    EARLY_STOP_PATIENCE = 30
    EARLY_STOP_MIN_DELTA = 1e-4


# ============================================================================
# MLP 모델 하이퍼파라미터 (레거시 - 사용 안 함)
# ============================================================================
class MLPConfig:
    """MLP 분류기 설정 (레거시)"""
    # 단일 부위 MLP (사용 안 함, 레거시)
    SINGLE_FEATURE_DIM = 192
    SINGLE_HIDDEN_DIMS = [128, 64]
    SINGLE_DROPOUT = 0.3
    
    # 멀티 부위 MLP (레거시 - Attention으로 교체됨)
    MULTI_FEATURE_DIM = 2880
    MULTI_HIDDEN_DIMS = [1024, 512, 256]
    MULTI_DROPOUT = [0.3, 0.2, 0.1]
    NUM_CLASSES = 2
    
    # 학습 설정
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 128
    EPOCHS = 200
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 15
    EARLY_STOP_MIN_DELTA = 1e-4


# ============================================================================
# Attention 모델 하이퍼파라미터 (현재 사용)
# ============================================================================
class AttentionConfig:
    """Attention 기반 분류기 설정"""
    # Attention 구조
    EMBED_DIM = 256  # 부위별 Feature를 통일할 차원
    SCORE_EMBED_DIM = 16  # Anomaly Score 임베딩 차원 (Feature: 240, Score: 16)
    NUM_HEADS = 8  # Multi-Head Attention head 수
    NUM_ENCODER_LAYERS = 2  # Transformer Encoder 레이어 수
    DROPOUT = 0.2  # Dropout 비율
    NUM_CLASSES = 2  # Normal, Abnormal
    
    # 학습 설정
    LEARNING_RATE = 5e-4  # Attention은 보통 낮은 LR 사용
    BATCH_SIZE = 256
    EPOCHS = 50
    
    # 최적화 설정
    WEIGHT_DECAY = 1e-4  # Weight decay
    GRAD_CLIP = 1.0  # Gradient clipping (Transformer 안정화)
    
    # 조기 종료 설정
    EARLY_STOP_PATIENCE = 20  # Attention은 수렴이 느릴 수 있음
    EARLY_STOP_MIN_DELTA = 1e-4
    
    # Warmup (선택사항)
    WARMUP_EPOCHS = 5  # 초기 LR을 점진적으로 증가
    
    # 클래스 불균형 대응 설정
    USE_FOCAL_LOSS = False  # Focal Loss 사용 여부
    FOCAL_GAMMA = 2.0  # Focal Loss gamma 값 (2.0 → 3.0: 어려운 샘플에 더 집중)
    USE_CLASS_WEIGHTS = False  # Class weights 사용 여부
    
    # 클래스 가중치 (Precision 향상을 위해 조정)
    # False Positive를 줄이려면 Normal 가중치를 높임
    NORMAL_WEIGHT = 1.0  # 정상 클래스 가중치 (1.0 → 2.0: 정상을 더 정확히)
    ABNORMAL_WEIGHT = 1.0  # 이상 클래스 가중치 (10.0 → 8.0: 균형 조정)


# ============================================================================
# 학습 일반 설정
# ============================================================================
class TrainConfig:
    """학습 일반 설정"""
    # 시드
    SEED = 42
    
    # 디바이스
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # 데이터 로더 설정
    NUM_WORKERS = 12
    PIN_MEMORY = True  # GPU로 데이터 전송 속도 향상
    PREFETCH_FACTOR = 4  # 각 워커가 미리 로드할 배치 수
    PERSISTENT_WORKERS = True  # 워커 프로세스 재사용
    
    # 로깅
    LOG_INTERVAL = 10  # 배치마다 로그 출력 간격
    
    # 체크포인트 저장
    SAVE_BEST_ONLY = True  # 최고 성능 모델만 저장
    SAVE_LAST = True  # 마지막 에포크 모델도 저장
    
    # 평가 설정
    EVAL_THRESHOLD = 0.5  # 이진 분류 임계값 (0.5 → 0.7: precision 향상, recall 약간 감소)


# ============================================================================
# 전체 설정을 하나로 묶은 클래스
# ============================================================================
class Config:
    """모든 설정을 포함하는 통합 클래스"""
    Path = PathConfig
    Data = DataConfig
    Joint = JointConfig
    STGNF = STGNFConfig
    MLP = MLPConfig
    Attention = AttentionConfig  # Attention 설정 추가
    Train = TrainConfig
    
    @classmethod
    def print_config(cls):
        """설정 출력"""
        print("=" * 80)
        print("학습 설정")
        print("=" * 80)
        print(f"시드: {cls.Train.SEED}")
        print(f"디바이스: {cls.Train.DEVICE}")
        print(f"학습할 부위: {cls.Joint.BODY_PARTS}")
        print(f"\n[데이터 설정]")
        print(f"  시퀀스 길이: {cls.Data.SEG_LEN}")
        print(f"  학습 stride: {cls.Data.TRAIN_STRIDE}")
        print(f"  평가 stride: {cls.Data.EVAL_STRIDE}")
        print(f"  데이터 증강: {cls.Data.APPLY_AUGMENTATION}")
        print(f"\n[STG-NF 설정]")
        print(f"  배치 크기: {cls.STGNF.BATCH_SIZE}")
        print(f"  에포크: {cls.STGNF.EPOCHS}")
        print(f"  학습률: {cls.STGNF.LEARNING_RATE}")
        print(f"  Hidden Dim: {cls.STGNF.HIDDEN_DIM}")
        print(f"\n[MLP 설정]")
        print(f"  배치 크기: {cls.MLP.BATCH_SIZE}")
        print(f"  에포크: {cls.MLP.EPOCHS}")
        print(f"  학습률: {cls.MLP.LEARNING_RATE}")
        print(f"  특징 차원: {cls.MLP.MULTI_FEATURE_DIM}")
        print(f"  Hidden Dims: {cls.MLP.MULTI_HIDDEN_DIMS}")
        print("=" * 80)
    
    @classmethod
    def from_args(cls, args=None):
        """
        커맨드라인 인자로부터 설정 업데이트
        나중에 argparse로 확장 가능
        """
        if args is None:
            return cls
        
        # 예시: args로부터 설정 업데이트
        if hasattr(args, 'seed'):
            cls.Train.SEED = args.seed
        if hasattr(args, 'device'):
            cls.Train.DEVICE = args.device
        if hasattr(args, 'batch_size'):
            cls.STGNF.BATCH_SIZE = args.batch_size
            cls.MLP.BATCH_SIZE = args.batch_size
        if hasattr(args, 'epochs'):
            cls.STGNF.EPOCHS = args.epochs
            cls.MLP.EPOCHS = args.epochs
        
        return cls


# ============================================================================
# Argparse 함수 (커맨드라인 인자 파싱)
# ============================================================================
def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='STG-NF + Attention 학습 파이프라인')
    
    # 학습 일반 설정
    parser.add_argument('--seed', type=int, default=TrainConfig.SEED,
                        help='랜덤 시드')
    parser.add_argument('--device', type=str, default=TrainConfig.DEVICE,
                        choices=['cuda', 'cpu'], help='학습 디바이스')
    
    # 데이터 설정
    parser.add_argument('--seg_len', type=int, default=DataConfig.SEG_LEN,
                        help='시퀀스 길이')
    parser.add_argument('--train_stride', type=int, default=DataConfig.TRAIN_STRIDE,
                        help='학습 시 stride')
    parser.add_argument('--eval_stride', type=int, default=DataConfig.EVAL_STRIDE,
                        help='평가 시 stride')
    
    # STG-NF 학습 설정
    parser.add_argument('--stgnf_batch_size', type=int, default=STGNFConfig.BATCH_SIZE,
                        help='STG-NF 배치 크기')
    parser.add_argument('--stgnf_epochs', type=int, default=STGNFConfig.EPOCHS,
                        help='STG-NF 에포크 수')
    parser.add_argument('--stgnf_lr', type=float, default=STGNFConfig.LEARNING_RATE,
                        help='STG-NF 학습률')
    
    # MLP 학습 설정
    parser.add_argument('--mlp_batch_size', type=int, default=MLPConfig.BATCH_SIZE,
                        help='MLP 배치 크기')
    parser.add_argument('--mlp_epochs', type=int, default=MLPConfig.EPOCHS,
                        help='MLP 에포크 수')
    parser.add_argument('--mlp_lr', type=float, default=MLPConfig.LEARNING_RATE,
                        help='MLP 학습률')
    
    # 부위 선택
    parser.add_argument('--parts', nargs='+', default=JointConfig.BODY_PARTS,
                        choices=['head', 'arms', 'body', 'legs', 'all'],
                        help='학습할 부위 선택')
    
    # 경로 설정
    parser.add_argument('--checkpoint_dir', type=str, default=PathConfig.CHECKPOINT_DIR,
                        help='체크포인트 저장 경로')
    parser.add_argument('--results_dir', type=str, default=PathConfig.RESULTS_DIR,
                        help='결과 저장 경로')
    
    return parser.parse_args()


if __name__ == '__main__':
    # 설정 출력 테스트
    Config.print_config()
    
    print("\n" + "=" * 80)
    print("경로 설정")
    print("=" * 80)
    print(f"BASE_DIR: {PathConfig.BASE_DIR}")
    print(f"CHECKPOINT_DIR: {PathConfig.CHECKPOINT_DIR}")
    print(f"TRAIN_JSON: {PathConfig.TRAIN_JSON}")
    print(f"MLP_TRAIN_JSON: {PathConfig.MLP_TRAIN_JSON}")
    print(f"TEST_JSON: {PathConfig.TEST_JSON}")
    print("=" * 80)
