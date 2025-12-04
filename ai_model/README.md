# Unmanned Store Shoplifting Detection (ai_model)

## 프로젝트 개요

이 프로젝트는 무인매장에서 발생하는 도난(이상행동) 탐지를 위해 STG-NF 기반 정상 패턴 모델과 Attention 분류기를 활용합니다. Skeleton 데이터 기반으로 정상/이상 행동을 분류하며, 전체 파이프라인은 다음과 같이 구성됩니다:

1. **Skeleton 추출** (YOLO 기반)
2. **STG-NF 모델 학습** (정상 패턴)
3. **Attention 분류기 학습** (이상/정상 분류)
4. **평가 및 결과 분석**

---

## 폴더/파일 구조 및 역할

- `args.py` : 전체 설정(경로, 하이퍼파라미터 등) 관리
- `train_stgnf.py` : STG-NF 모델 학습 스크립트
- `train_attention.py` : Attention 분류기 학습 스크립트
- `eval_stgnf.py` : STG-NF 모델 평가 스크립트
- `eval_pipeline.py` : 전체 파이프라인 평가 자동화
- `datasets/` : 데이터셋 클래스 및 학습 유틸리티
- `extract_skeleton/` : skeleton 데이터 추출 및 검증
- `models/` : 모델 생성/로딩 및 STG-NF 구현
- `checkpoints/` : 학습된 모델 가중치 및 로그
- `results/` : 평가 결과 저장

---

## 사용법

### 1. 학습 환경 구성

#### 개발 환경 사양
- **CPU**: AMD Ryzen 9 9950X (16-Core, 32-Thread)
- **RAM**: 64GB DDR5
- **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
- **OS**: Windows 11
- **CUDA**: 12.9
- **Python**: 3.10

> **참고**: RTX 5080과 같은 최신 GPU는 CUDA 12.9가 필요합니다.

#### Conda 환경 설치 (권장)

**Option 1: environment.yml 사용 (권장)**
```bash
# Conda 환경 생성
conda env create -f ai_model/environment.yml

# 환경 활성화
conda activate shoplifting_train

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Option 2: requirements_full.txt 사용**
```bash
# Conda 환경 생성
conda create -n shoplifting_train python=3.10 -y
conda activate shoplifting_train

# 패키지 설치
pip install -r requirements_full.txt

# PyTorch CUDA 버전 설치 (RTX 5080 지원)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
```

#### GPU 확인
```bash
# NVIDIA GPU 확인
nvidia-smi

# PyTorch CUDA 사용 가능 여부 확인
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 2. 데이터 준비 및 전처리

#### 데이터셋 구성
- **정상 데이터**: AI-Hub 실내(편의점, 매장) 구매행동 데이터
- **이상 데이터**: AI-Hub 실내(편의점, 매장) 이상행동 데이터 (**절도** 카테고리만)


#### 학습용 데이터 전처리

**1단계: 정상 데이터 Skeleton 추출 (STG-NF 학습용)**
```bash
cd ai_model/extract_skeleton

# XML bbox 기반 다중 인물 추출 (점주 제외, ID 있는 인물만)
python skeleton_extract.py \
    --video_dir /path/to/normal/videos \
    --xml_dir /path/to/normal/xmls \
    --output_dir ../data/train_stgnf/skeleton_data \
    --mode xml \
    --workers 4
```

**특징:**
- XML 파일에서 ID 속성이 있는 모든 인물 추출 (점주 등 ID 없는 인물 제외)
- 1개 영상에서 여러 인물의 skeleton 추출 가능
- YOLO11x-pose 모델 사용 (conf_threshold=0.5)
- 출력 형식: JSON (각 인물별 skeleton 시퀀스)

**2단계: 정상+이상 데이터 Skeleton 추출 (Attention 학습용)**
```bash
# 이상 데이터: 단순 1인 추출 (XML 없이)
python skeleton_extract.py \
    --video_dir /path/to/abnormal/videos \
    --output_dir ../data/train_attention/skeleton_data \
    --mode single \
    --workers 4

# 정상 데이터 일부 추가 (균형 맞추기)
python skeleton_extract.py \
    --video_dir /path/to/normal/videos \
    --output_dir ../data/train_attention/skeleton_data \
    --mode single \
    --workers 4
```

**특징:**
- 가장 크고 명확한 1명의 skeleton만 추출
- 이상 데이터는 XML 없이 단순 추출
- 정상/이상 데이터 비율: 약 1:1 권장 -> 이상 프레임 개수가 1:1로 해도 매우 적음

**3단계: Ground Truth 생성**
```bash
# GT 파일 생성 (train_attention_gt, test_gt)
# - 정상: 0, 이상: 1
# - .npy 형식으로 저장
python create_gt.py \
    --normal_list normal_videos.txt \
    --abnormal_list abnormal_videos.txt \
    --output_dir ../data/gt/train_attention_gt
```

#### 평가용 데이터 전처리

**테스트 데이터 Skeleton 추출**
```bash
cd ai_model/extract_skeleton

# 테스트 세트: 1인 추출
python skeleton_extract.py \
    --video_dir /path/to/test/videos \
    --output_dir ../data/test/skeleton_data \
    --mode single \
    --workers 4

# GT 생성
python create_gt.py \
    --normal_list test_normal.txt \
    --abnormal_list test_abnormal.txt \
    --output_dir ../data/gt/test_gt
```

#### 데이터 구조
```
data/
├── train_stgnf/
│   └── skeleton_data/       # 정상 행동 skeleton (JSON)
│       ├── video_001.json    # 다중 인물 가능
│       └── video_002.json
├── train_attention/
│   └── skeleton_data/       # 정상+이상 skeleton (JSON)
│       ├── normal_001.json   # 1인 skeleton
│       └── abnormal_001.json # 1인 skeleton
├── test/
│   └── skeleton_data/       # 평가용 skeleton (JSON)
└── gt/
    ├── train_attention_gt/  # 학습용 GT (.npy)
    └── test_gt/              # 평가용 GT (.npy)
```

#### Skeleton 데이터 형식
```json
{
    "video_name": "example.mp4",
    "fps": 3,
    "total_frames": 600,
    "person_id": [
        {
            "frame_idx": 0,
            "keypoints": [[x1, y1, conf1], [x2, y2, conf2], ...]  // 17개 관절
        },
        ...
    ]
}
```

#### 데이터 검증
```bash
# Skeleton 데이터 무결성 검증
cd ai_model/extract_skeleton
python verify_data.py \
    --skeleton_dir ../data/train_stgnf/skeleton_data \
    --check_min_frames 12  # 최소 12프레임 이상 확인
```

---

### 3. 모델 입력 전 데이터 전처리

Skeleton 추출 후 모델에 입력하기 전 5단계 전처리가 자동 수행됩니다:

#### 0단계: 목 관절 추가 (COCO17 → COCO18 변환)
```python
# 기능: YOLO가 출력하는 COCO17 포맷을 COCO18로 변환
# 방법: 양쪽 어깨(5,6번) 중간 지점을 Neck(1번) 관절로 추가
# 변환: neck = (left_shoulder + right_shoulder) / 2
# 재정렬: [0,17,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3] 순서로

coco18_skeleton = keypoints17_to_coco18(coco17_skeleton)
# COCO17: 17개 관절 (Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles)
# COCO18: 18개 관절 (+ Neck)
```

**효과:**
- STG-NF 모델이 요구하는 COCO18 포맷으로 변환
- Neck(1번) 관절을 Root-Relative 및 Scale Normalization의 기준점으로 사용
- 상체 중심점 확보로 정규화 안정성 향상

#### 1단계: 결측치 보간 (Interpolation)
```python
# 기능: 0 값(결측치)를 선형 보간으로 채움
# 처리: Pandas interpolate (linear, bidirectional)
# 예시:
#   원본: [10, 0, 0, 13, 14]
#   보간: [10, 11, 12, 13, 14]

interpolated_skeleton = interpolate_skeleton(skeleton_data)
```

**효과:**
- 프레임 누락으로 인한 데이터 손실 방지
- YOLO 감지 실패 프레임 보완
- 시계열 연속성 확보

#### 2단계: 상대 좌표 변환 (Root-Relative Normalization)
```python
# 기능: 목(Neck) 관절을 기준(0,0)으로 모든 관절 좌표 변환
# 방법: x_rel = x - x_neck, y_rel = y - y_neck
# 관절: COCO18 기준 1번(Neck)

relative_skeleton = convert_to_relative_coordinates(skeleton, root_joint_idx=1)
```

**효과:**
- 영상 내 인물 위치 불변성 확보
- 카메라 시점 변화 대응
- 모델이 자세/동작에만 집중

#### 3단계: 스케일 정규화 (Scale Normalization)
```python
# 기능: 몸통 길이 기반 스케일 통일
# 방법: 목-골반 거리의 중앙값으로 나눈
# 관절: Neck(1) - Hip(8)

normalized_skeleton = apply_scale_normalization(skeleton, neck_idx=1, hip_idx=8)
```

**효과:**
- 인물 크기 불변성 확보
- 카메라 거리 변화 대응
- OpenPose 좌표 떨림 노이즈 방지 (중앙값 사용)

#### 4단계: 데이터 증강 (Data Augmentation) - 학습 시만
```python
# 적용 조건: apply_augmentation=True (학습 시만)
# 변환 종류:
#   1. Identity: 변환 없음
#   2. Flip: 좌우 반전 (x → -x)
#   3. Shear: 전단 변환 (shearx=0.1, sheary=0.1)
#   4. Flip+Shear: 반전 + 전단 조합

augmented_skeleton = random.choice([transform_identity, transform_flip, 
                                   transform_shear, transform_flip_shear])(skeleton)
```

**효과:**
- 모델 강건성 향상
- 과적합 방지
- 훈련 데이터 다양성 확보

#### 전체 파이프라인
```
원본 Skeleton (17관절, COCO17, 30fps)
    ↓
0. 목 관절 추가 (COCO17 → COCO18, Neck=어깨 중간점)
    ↓
1. 결측치 보간 (0값 → 선형 보간)
    ↓
2. 상대 좌표 (Neck 기준 변환)
    ↓
3. 스케일 정규화 (몸통 길이 기준)
    ↓
4. 데이터 증강 (학습 시만, 반전/전단)
    ↓
24프레임 시퀀스 분할 (stride=12 or 1)
    ↓
STG-NF / Attention 모델 입력 (18관절)
```

#### 전처리 파라미터 (args.py)
```python
# 데이터 설정
SEG_LEN = 24              # 시퀀스 길이 (24프레임 = 0.8초)
TRAIN_STRIDE = 12         # 학습 stride (50% 겹침)
EVAL_STRIDE = 1           # 평가 stride (모든 프레임)
NORMALIZE = True          # 스케일 정규화 활성화
APPLY_AUGMENTATION = True # 데이터 증강 (학습 시)
VID_RES = [1920, 1080]    # 영상 해상도
```

---

### 4. 모델 학습 설정
- 모든 하이퍼파라미터 및 경로는 `ai_model/args.py`에서 관리
  - 시퀀스 길이, stride, 배치 크기, epoch 등 직접 수정 가능
  - 커맨드라인 인자도 지원 (예: `--epochs 100`)

#### 주요 하이퍼파라미터 (args.py)
```python
# 데이터 설정
SEG_LEN = 24              # 시퀀스 길이 (24프레임 = 약 0.8초)
TRAIN_STRIDE = 12         # 학습 stride (50% 겹침)
EVAL_STRIDE = 1           # 평가 stride (모든 프레임)

# STG-NF 설정
HIDDEN_CHANNELS = 64      # 은닉 채널 크기
BATCH_SIZE = 256          # 배치 크기 (GPU 메모리에 따라 조정)
EPOCHS = 50               # 학습 에포크

# Attention 설정
EMBED_DIM = 256           # 특징 임베딩 차원
NUM_HEADS = 4             # Attention Head 수
DROPOUT = 0.5             # Dropout 비율
```

### 4. STG-NF 모델 학습
```bash
python train_stgnf.py
```
- 정상 데이터만 사용하여 STG-NF 모델을 학습합니다.
- 체크포인트는 `checkpoints/`에 저장됩니다.

### 5. Attention 분류기 학습
```bash
python train_attention.py
```
- STG-NF에서 추출한 특징을 활용해 이상/정상 분류기를 학습합니다.
- Attention 모델 가중치가 `checkpoints/attention_fin.pth`에 저장됩니다.

### 6. 평가
#### (1) STG-NF 평가
```bash
python eval_stgnf.py
```
- 프레임 단위로 정상/이상 판별 성능을 평가합니다.

#### (2) 전체 파이프라인 평가
```bash
python eval_pipeline.py
```
- STG-NF + Attention 분류기 조합의 최종 성능을 평가합니다.
- 결과는 `results/` 폴더에 저장됩니다.

---

## 주요 설정 위치
- **모델/데이터/학습 설정**: `ai_model/args.py`
  - 직접 수정하거나 커맨드라인 인자로 전달 가능
- **데이터 경로**: `args.py`의 `PathConfig` 클래스에서 관리
- **하이퍼파라미터**: `DataConfig`, `STGNFConfig`, `AttentionConfig` 등에서 관리

---

## 참고 사항

### 학습 소요 시간 (RTX 5080 기준)
- **STG-NF 학습**: 부위당 약 30-60분 (총 5개 부위 = 약 2.5-5시간)
- **Attention 학습**: 약 10-20분
- **전체 파이프라인**: 초기 학습 약 3-6시간
- 위 시간은 RAM 용량이 충분할 시 가능(pin_memory 및 RAM에 데이터 적재하는 방식으로 학습 진행)

### 시스템 요구사항
- **GPU**: NVIDIA RTX 3060 이상 권장 (VRAM 12GB+)
- **RAM**: 16GB 이상 (32GB 권장)
- **저장 공간**: 50GB 이상 (데이터셋 + 체크포인트)
- **CUDA**: 11.8 이상 (RTX 5080의 경우 12.9 필수)

### 문제 해결
- **CUDA Out of Memory**: `args.py`에서 `BATCH_SIZE` 감소 (256 → 128)
- **수렴 안 됨**: Learning rate 조정 또는 Epoch 증가
- **Import 오류**: `PYTHONPATH` 설정 확인 (`export PYTHONPATH=/path/to/project`)
- **데이터 로딩 실패**: `args.py`의 경로 설정 확인

### 추가 정보
- Skeleton 데이터는 `extract_skeleton/skeleton_extract.py`로 생성
- 모델 학습/평가 시 GPU 사용 권장 (`args.py`에서 `DEVICE` 설정)
- 각 스크립트 실행 전 환경 및 경로 설정을 반드시 확인하세요
- 학습된 모델은 `checkpoints/` 폴더에 자동 저장됩니다

---

