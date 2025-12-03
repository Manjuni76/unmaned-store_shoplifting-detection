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
unmaned_store_shoplifting_detection/
│
├── ai_model/              # AI 모델 학습 및 평가
│   ├── args.py            # 전체 설정 (하이퍼파라미터, 경로)
│   ├── train_stgnf.py     # STG-NF 학습
│   ├── train_attention.py # Attention 분류기 학습
│   ├── eval_stgnf.py      # STG-NF 평가
│   ├── eval_pipeline.py   # 전체 파이프라인 평가
│   ├── checkpoints/       # 학습된 모델 가중치
│   ├── datasets/          # 데이터셋 로더
│   ├── models/            # 모델 구조 정의
│   └── extract_skeleton/  # YOLO skeleton 추출
│
├── ai_server/             # 추론 서버 (배포용)
│   ├── celery_app.py      # Celery 설정
│   ├── tasks.py           # 비동기 작업 처리
│   └── inference.py       # 실제 추론 로직
│
├── backend/               # 웹 서버
│   ├── main.py            # FastAPI 서버
│   ├── database.py        # SQLite DB
│   └── templates/         # HTML UI
│
└── data/                  # 학습/평가 데이터
    ├── train_stgnf/       # STG-NF 학습용 (정상만)
    ├── train_attention/   # Attention 학습용 (정상+이상)
    ├── test/              # 평가용
    └── gt/                # Ground Truth 라벨


---

## 빠른 시작

### 1. 모델 학습 (ai_model/)
```bash
# 환경 활성화
conda activate your_env

# STG-NF 학습 (부위별)
cd ai_model
python train_stgnf.py

# Attention 분류기 학습
python train_attention.py

# 평가
python eval_pipeline.py
```


### 2. 서버 배포 (Docker)
```bash
# Docker Compose로 한 번에 실행
docker-compose up --build

# 브라우저에서 접속
http://localhost:8001
```

자세한 내용: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 데이터 준비

### 학습 데이터 구조
```
data/
├── train_stgnf/skeleton_data/       # 정상 행동 skeleton (JSON)
│   ├── video_001.json
│   └── ...
├── train_attention/skeleton_data/   # 정상+이상 skeleton
│   └── ...
├── test/skeleton_data/              # 평가용 skeleton
│   └── ...
└── gt/                              # Ground Truth
    ├── train_attention_gt/          # 학습용 GT (.npy)
    └── test_gt/                     # 평가용 GT (.npy)
```

### Skeleton 추출
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
| 메트릭 | Train | Val |
|--------|-------|-----|
| Accuracy | 94.5% | 92.8% |
| F1 Score | 93.2% | 91.5% |
| Precision | 92.1% | 90.3% |
| Recall | 94.3% | 92.7% |

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
- **Anomaly Score Embedding**: STG-NF 이상 점수를 임베딩으로 활용
- **Focal Loss**: 클래스 불균형 대응

---

## 개발 환경

### 필수 요구사항
- Python 3.8+
- PyTorch 2.1+
- CUDA 11.8+ (GPU 사용 시)
- Redis (배포 시)
- Docker + NVIDIA Docker (배포 시)

### 패키지 설치
```bash
# 학습 환경
pip install torch torchvision ultralytics opencv-python numpy scikit-learn tqdm matplotlib seaborn

# 배포 환경 (추가)
pip install fastapi uvicorn celery redis boto3
```

---

## 사용법

### 1. 모델 학습
```bash
# STG-NF 학습
python ai_model/train_stgnf.py

# Attention 학습
python ai_model/train_attention.py
```

### 2. 모델 평가
```bash
# 전체 파이프라인 평가
python ai_model/eval_pipeline.py
```

### 3. 웹 서비스 실행
```bash
# Docker로 실행
docker-compose up

# 또는 로컬 실행
# Terminal 1: Redis
redis-server

# Terminal 2: Backend
uvicorn backend.main:app --reload

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

### 학습 관련
- **메모리 부족**: `BATCH_SIZE` 감소
- **수렴 안 됨**: `LEARNING_RATE` 조정, `EPOCHS` 증가
- **과적합**: `DROPOUT` 증가, 데이터 증강 활성화

### 배포 관련
- **모델 로딩 실패**: `ai_model/checkpoints/` 경로 확인
- **GPU 인식 안 됨**: NVIDIA Docker 설치 확인
- **Import 오류**: `PYTHONPATH` 환경 변수 확인

---

## 프로젝트 팀
- AI 모델 개발: STG-NF + Attention Classifier
- 데이터 전처리: YOLO11 skeleton 추출
- 백엔드 개발: FastAPI + Celery
- 배포: Docker + AWS S3

---

## 라이선스
이 프로젝트는 교육 목적으로 개발되었습니다.

---

## 참고 문헌
- STG-NF: "STG-NF: Spatial-Temporal Graph Normalizing Flow for Video Anomaly Detection"
- YOLOv11: "Ultralytics YOLO11 Pose Estimation"
- Attention Mechanism: "Attention Is All You Need"

---

## 추가 문서
- [ai_model/README.md](ai_model/README.md) - 모델 학습 가이드
- [DEPLOYMENT.md](DEPLOYMENT.md) - 배포 가이드
- [ai_model/args.py](ai_model/args.py) - 설정 파일
