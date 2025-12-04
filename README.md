# 무인매장 도난 탐지 시스템 (Unmanned Store Shoplifting Detection)

## 프로젝트 개요
부위 별 STG-NF(Spatial-Temporal Graph Normalizing Flow)모델과 Attention Classifier를 결합한 skeleton pose 기반 도난 행동 탐지 시스템입니다.

### 주요 특징
- **STG-NF**: 정상 패턴만 학습하는 비지도 학습 방식
- **부위별 분석**: 머리, 팔, 몸, 다리, 전체 5개 부위 별 특징 학습
- **Attention Fusion**: 부위별 특징을 Attention으로 통합하여 최종 판별
- **프레임 단위 탐지**: 도난 발생 정확한 시점(초) 추출
- **실시간 배포**: FastAPI + Celery + Docker 기반 운영 시스템

---

## 시스템 아키텍처

```
[영상 업로드] → [Skeleton 추출(YOLO11)] → [유효 구간 필터링(12프레임 이상)]
                                                      ↓
[웹 UI 결과 표시] ← [도난 구간 반환] ← [Attention 분류] ← [STG-NF 특징 추출(5개 부위)]
```

---

## 폴더 구조
```text
unmaned_store_shoplifting_detection/
│
├── ai_model/                  # AI 모델 학습 및 평가 관련 코드
│   ├── checkpoints/           # 학습된 모델 가중치 (.pth 파일)
│   ├── datasets/              # 데이터셋 로더 및 전처리
│   ├── extract_skeleton/      # YOLOv11 기반 스켈레톤 추출 로직
│   ├── models/                # STG-NF 및 Attention 모델 아키텍처 정의
│   ├── args.py                # 전체 설정 관리 (하이퍼파라미터, 경로 등)
│   ├── eval_pipeline.py       # 전체 파이프라인 성능 평가 스크립트
│   ├── eval_stgnf.py          # STG-NF 단독 성능 평가
│   ├── train_attention.py     # Attention 분류기 학습 스크립트
│   └── train_stgnf.py         # STG-NF 모델 학습 스크립트
│
├── ai_server/                 # AI 추론 서버 (Celery Worker)
│   ├── celery_app.py          # Celery 애플리케이션 설정
│   ├── inference.py           # 핵심 추론 로직 (모델 로드 및 예측)
│   └── tasks.py               # 비동기 작업(Task) 정의
│
├── backend/                   # 웹 서버 (FastAPI)
│   ├── templates/             # 웹 UI (HTML/JS)
│   ├── database.py            # SQLite 데이터베이스 설정
│   └── main.py                # FastAPI 엔트리포인트 및 API 라우터
│
├── data/                      # 데이터셋 디렉토리
│   ├── gt/                    # Ground Truth 라벨 파일
│   ├── test/                  # 평가용 비디오 및 스켈레톤 데이터
│   ├── train_attention/       # Attention 모델 학습용 데이터
│   └── train_stgnf/           # STG-NF 모델 학습용 데이터 (정상 구간)
│
├── docker-compose.yml         # 전체 서비스 실행 설정 (Docker Compose)
├── Dockerfile                 # 도커 이미지 빌드 설정
├── requirements.txt           # 파이썬 의존성 패키지 목록
└── README.md                  # 프로젝트 문서
```

---

## 모델 학습 실행 방법

### 1. 환경 요구사항
- **Python**: 3.10 이상
- **PyTorch**: CPU 버전 (Docker 이미지에 포함)
- **RAM**: 8GB 이상 권장
- **저장 공간**: 10GB 이상

> **참고 사항**: 프로젝트에서 사용한 GPU 5080에 맞는 cuda 설정 배포시 오류가 발생하여 CPU 버전으로 배포합니다. 처리 속도는 1분 영상 기준 약 20초 정도 차이 났습니다(RYZEN 9 9950X 기준)

### 2. 모델 학습 (ai_model/)
```bash
# 환경 활성화
conda activate your_env

# STG-NF 학습 (부위별)
cd ai_model
python train_stgnf.py

# Attention 분류기 학습
cd ai_model
python train_attention.py

# STG-NF 평가
cd ai_model
python eval_stgnf.py

# STG-NF + Attention 평가
cd ai_model
python eval_pipeline.py
```

### 3. 서버 배포 (Docker)
```bash
# Docker Compose로 한 번에 실행
docker-compose up --build

# 브라우저에서 접속
http://localhost:8001
```

자세한 내용: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 데이터 준비 및 전처리

### 데이터셋 출처
- **정상 데이터**: AI-Hub 실내(편의점, 매장) 구매행동 데이터
- **이상 데이터**: AI-Hub 실내(편의점, 매장) 이상행동 데이터 (**절도** 카테고리만)
- **총 데이터**: 약 10,000개 영상 (2TB)

---

### 학습 데이터 전처리

#### 1단계: 정상 데이터 Skeleton 추출 (STG-NF 학습용)
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
- YOLO11x-pose 모델 사용 (17개 관절점)

#### 2단계: 이상 데이터 Skeleton 추출 (Attention 학습용)
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
- 정상/이상 데이터 비율: 약 1:1 권장

#### 3단계: Ground Truth 생성
```bash
# GT 파일 생성 (.npy 형식)
python create_gt.py \
    --normal_list normal_videos.txt \
    --abnormal_list abnormal_videos.txt \
    --output_dir ../data/gt/train_attention_gt
```

---

### 평가 데이터 전처리

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

---

### 배포 시 실시간 데이터 전처리

**웹 서비스 실행 시 자동 처리:**

1. **영상 업로드** → S3에 저장
2. **Celery Worker가 자동 처리:**
   ```python
   # ai_server/inference.py에서 실시간 처리
   
   # Step 1: YOLO11x-pose로 Skeleton 추출
   skeleton_data = skeleton_extractor.extract_skeleton_single_person(video_path)
   
   # Step 2: 12프레임 이상 유효 구간만 선별
   valid_segments = filter_valid_segments(skeleton_data, min_frames=12)
   
   # Step 3: Skeleton 정규화 (1920x1080 기준)
   normalized_skeleton = normalize_skeleton(skeleton_data)
   
   # Step 4: 24프레임 시퀀스로 분할 (stride=1)
   sequences = create_sequences(normalized_skeleton, seq_len=24, stride=1)
   
   # Step 5: STG-NF + Attention 모델로 추론
   predictions = model.predict(sequences)
   ```

3. **결과 반환:** 도난 의심 구간 + 위험도 레벨

**처리 속도:**
- CPU 버전: 1분 영상 약 20초 (Ryzen 9 9950X 기준)
- GPU 버전: 1분 영상 약 5초 (RTX 5080 기준)

---

### 학습 데이터 구조
```
data/
├── train_stgnf/
│   └── skeleton_data/       # 정상 행동 skeleton (JSON)
│       ├── video_001.json   # 다중 인물 가능
│       └── video_002.json
├── train_attention/
│   └── skeleton_data/       # 정상+이상 skeleton (JSON)
│       ├── normal_001.json  # 1인 skeleton
│       └── abnormal_001.json # 1인 skeleton
├── test/
│   └── skeleton_data/       # 평가용 skeleton (JSON)
│       └── test_001.json
└── gt/                      # Ground Truth
    ├── train_attention_gt/  # 학습용 GT (.npy, 0=정상, 1=이상)
    └── test_gt/             # 평가용 GT (.npy)
```

### Skeleton JSON 형식
```json
{
    "video_name": "example.mp4",
    "fps": 30,
    "total_frames": 900,
    "skeletons": [
        {
            "frame_idx": 0,
            "keypoints": [[x1, y1, conf1], [x2, y2, conf2], ...]  // 17개 관절점
        }
    ]
}
```

### 데이터 전처리 요약

| 단계 | 입력 | 출력 | 도구 | 특징 |
|------|------|------|------|------|
| **STG-NF 학습용** | 정상 영상 + XML | 다중 인물 skeleton JSON | `skeleton_extract.py --mode xml` | ID 있는 인물만 추출 |
| **Attention 학습용** | 정상+이상 영상 | 1인 skeleton JSON | `skeleton_extract.py --mode single` | 가장 큰 인물 1명 |
| **테스트용** | 테스트 영상 | 1인 skeleton JSON | `skeleton_extract.py --mode single` | 평가 데이터 |
| **배포 실시간** | 업로드 영상 | 1인 skeleton | `inference.py` 자동 처리 | 12프레임 이상 필터링 |

---

### 모델 입력 전 데이터 전처리

Skeleton 추출 후 모델에 입력하기 전 5단계 전처리가 자동 수행됩니다:

#### 0. 목 관절 추가 (COCO17 → COCO18 변환)
- **목적**: YOLO 출력(COCO17)을 STG-NF 포맷(COCO18)으로 변환
- **방법**: 양쪽 어깨(5,6번) 중간점을 Neck(1번) 관절로 추가
  - neck = (left_shoulder + right_shoulder) / 2
  - 재정렬: [0,17,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3] -> openpose 18 형태
- **효과**: 상체 중심점 확보, Root-Relative 및 Scale Normalization 기준점

#### 1. 결측치 보간 (Interpolation)
- **목적**: YOLO 감지 실패 프레임 보완
- **방법**: 0 값(결측치)을 선형 보간으로 채움 (Pandas interpolate)
- **효과**: 시계열 연속성 확보, 데이터 손실 방지

#### 2. 상대 좌표 변환 (Root-Relative Normalization)
- **목적**: 인물 위치 불변성 확보
- **방법**: 목(Neck) 관절을 기준(0,0)으로 모든 관절 변환
  - x_rel = x - x_neck
  - y_rel = y - y_neck
- **효과**: 카메라 시점 변화 대응, 모델이 자세/동작에만 집중

#### 3. 스케일 정규화 (Scale Normalization)
- **목적**: 인물 크기 불변성 확보
- **방법**: 몸통 길이(목-골반 거리)의 중앙값으로 나눈
  - scale = median(||Neck - Hip||) 각 프레임에서
  - normalized = skeleton / scale
- **효과**: 카메라 거리 변화 대응, OpenPose 떨림 노이즈 방지

#### 4. 데이터 증강 (Data Augmentation) - 학습 시만
- **목적**: 모델 강건성 향상, 과적합 방지
- **변환 종류**:
  - Identity: 변환 없음
  - Flip: 좌우 반전 (x → -x)
  - Shear: 전단 변환 (shearx=0.1, sheary=0.1)
  - Flip+Shear: 반전 + 전단 조합
- **적용**: 학습 시에만 랜덤 적용 (평가/배포 시 X)

#### 전체 파이프라인
```
원본 Skeleton (17관절, COCO17, 30fps)
    ↓
목 관절 추가 (COCO17 → COCO18)
    ↓
결측치 보간 (0값 → 선형 보간)
    ↓
상대 좌표 (Neck 기준)
    ↓
스케일 정규화 (몸통 길이)
    ↓
[학습 시] 데이터 증강
    ↓
24프레임 시퀀스 분할
    ↓
STG-NF / Attention 모델 (18관절)
```

---

### Skeleton 추출 (선택사항)

```bash
cd ai_model/extract_skeleton
python skeleton_extract.py \
    --video_path /path/to/video.mp4 \
    --output_path /path/to/output.json \
    --mode single  # 단일 인물 추출
```

---

## 모델 성능

### STG-NF (부위별)
| 부위 | AUC-ROC | AUC-PR | EER |
|------|---------|--------|-----|
| Head | 85.2%   | 82.1%  | 20.3% |
| Arms | 88.7%   | 85.4%  | 18.5% |
| Body | 87.3%   | 84.2%  | 19.1% |
| Legs | 86.5%   | 83.8%  | 19.7% |
| All  | 89.1%   | 86.3%  | 17.9% |

### Attention Classifier (최종)
| AUC-ROC | AUC-PR | EER |
|--------|-------|-----|
| 0.9732 | 0.7020 | 0.073 |


---

## 주요 기술

### STG-NF (Spatial-Temporal Graph Normalizing Flow)
- **Normalizing Flow**: 정상 패턴의 확률 분포 학습
- **Graph Convolution**: skeleton 관절 간 공간적 관계 모델링
- **Temporal Convolution**: 시간축 움직임 패턴 학습
- **One-Class Learning**: 정상 데이터만으로 학습, 이상 탐지

### Attention Classifier
- **Multi-Head Attention**: 부위별 특징의 중요도 학습
- **Feature Fusion**: 5개 부위 특징을 효과적으로 통합
- **Anomaly Score Embedding**: STG-NF의 잠재벡터 z를 임베딩으로 활용
- **Focal Loss**: 클래스 불균형 대응

---

## 개발 환경

### 학습 환경 (로컬 GPU 필요)
**요구사항:**
- Python 3.10+
- PyTorch 2.8.0 (CUDA 12.9 포함)
- NVIDIA GPU (RTX 5080 사용)
- RAM: 16GB 이상
- VRAM: 12GB 이상

**패키지 설치:**
```bash
# requirements_full.txt 사용 (GPU 버전)
pip install -r requirements_full.txt

# 또는 수동 설치
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
pip install ultralytics opencv-python numpy scikit-learn tqdm matplotlib seaborn
```

### 배포 환경 (CPU만 사용)
**요구사항:**
- Docker Desktop
- RAM: 8GB 이상 (16GB 권장)
- CPU: 멀티코어 프로세서
- 저장 공간: 10GB 이상

**특징:**
- GPU 없이 실행 가능 (PyTorch CPU 버전)
- 처리 속도: 1분 영상 약 40초 소요 (Ryzen 9 9950X 기준)
- 정확도는 GPU 버전과 동일

**패키지 설치:**
```bash
# requirements.txt 사용 (CPU 버전, Docker에서 자동 설치)
pip install -r requirements.txt
```

> **참고:** 
> - `requirements_full.txt`: GPU 학습용 (PyTorch CUDA 12.9 포함)
> - `requirements.txt`: CPU 배포용 (Docker에서 사용)

---

## 프로젝트 실행 방법

### 1. 모델 학습 (로컬 GPU 환경)
```bash
# 1. GPU 환경 설정
conda create -n shoplifting_train python=3.10
conda activate shoplifting_train
pip install -r requirements_full.txt

# 2. STG-NF 학습 (부위별)
cd ai_model
python train_stgnf.py

# 3. Attention 분류기 학습
python train_attention.py

# 4. 모델 평가
python eval_pipeline.py
```

### 2. 웹 서비스 배포 (Docker CPU 환경)
```bash
# Docker로 실행 (권장)
docker-compose up --build

# 브라우저 접속
http://localhost:8001
```

**또는 로컬 실행 (개발 모드):**
```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Backend
cd backend
uvicorn main:app --reload --port 8001

# Terminal 3: AI Worker
celery -A ai_server.celery_app worker --loglevel=info --pool=solo
```

---

## 설정 변경

모든 하이퍼파라미터는 `ai_model/args.py`에서 관리합니다.

### 주요 설정
```python
# 데이터 설정
SEG_LEN = 24              # 시퀀스 길이 (24프레임 = 약 0.8초)
TRAIN_STRIDE = 12         # 학습 stride (50% 겹침)
EVAL_STRIDE = 1           # 평가 stride (모든 프레임)

# STG-NF 설정
HIDDEN_CHANNELS = 64      # 모델 크기
BATCH_SIZE = 256          # 배치 크기
EPOCHS = 50               # 학습 에포크

# Attention 설정
EMBED_DIM = 256           # 특징 임베딩 차원
NUM_HEADS = 4             # Attention Head 수
DROPOUT = 0.5             # Dropout 비율
```

---

## 문제 해결

### 학습 관련 (로컬 GPU)
- **메모리 부족**: `BATCH_SIZE` 감소 (256 → 128)
- **CUDA 오류**: PyTorch와 CUDA 버전 일치 확인
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- **수렴 안 됨**: `LEARNING_RATE` 조정 (1e-4 → 1e-3), `EPOCHS` 증가
- **과적합**: `DROPOUT` 증가 (0.5 → 0.6), 데이터 증강 활성화

### 배포 관련 (Docker CPU)
- **모델 로딩 실패**: `ai_model/checkpoints/` 폴더에 `.pth` 파일 확인
- **Docker 실행 안 됨**: Docker Desktop이 실행 중인지 확인
- **처리 속도 느림**: 정상 동작 (CPU 버전은 1분 영상에 약 20초 소요)
- **Import 오류**: `PYTHONPATH=/app` 환경 변수 확인 (Docker 내부에서 자동 설정됨)

---

## 라이선스
이 프로젝트는 교육 목적으로 개발되었습니다.

---

## 참고 문헌
- STG-NF: "Normalizing Flows for Human Pose Anomaly Detection", (O Hirschorn, ICCV23)
Source: https://github.com/orhir/STG-NF?tab=readme-ov-file

- YOLOv11: "Ultralytics YOLO11 Pose Estimation" @software
{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}

- Attention Mechanism: "Attention Is All You Need"

---

## 패키지 파일 설명

### requirements.txt (CPU 배포용)
- **용도**: Docker 배포 환경에서 사용
- **특징**: PyTorch CPU 버전 (GPU 미포함)
- **크기**: 약 1.5GB
- **설치**: Docker 빌드 시 자동 설치

### requirements_full.txt (GPU 학습용)
- **용도**: 로컬 GPU 환경에서 모델 학습
- **특징**: PyTorch 2.8.0 + CUDA 12.9 (RTX 5080 지원)
- **크기**: 약 6GB
- **설치**: `pip install -r requirements_full.txt`

> **주의**: RTX 5080 같은 최신 GPU는 CUDA 12.9가 필요하여 `requirements_full.txt` 사용 필수

---

## 추가 문서
- [ai_model/README.md](ai_model/README.md) - 모델 학습 가이드
- [DEPLOYMENT.md](DEPLOYMENT.md) - 배포 가이드
- [ai_model/args.py](ai_model/args.py) - 설정 파일
