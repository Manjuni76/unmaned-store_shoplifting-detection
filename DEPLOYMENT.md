# 무인매장 도난 탐지 시스템 - Docker 배포 가이드

## 프로젝트 개요
STG-NF + Attention 기반 도난 탐지 AI 시스템을 Docker로 배포합니다.

### 주요 기능
1. 영상 업로드 (쇼핑 장면)
2. YOLO11 기반 skeleton 추출
3. 12프레임 이상 사람이 있는 구간만 도난 탐지 진행
4. STG-NF + Attention 모델로 도난 구간 판별
5. 도난 의심 구간 표시 및 결과 반환

---

## 폴더 구조
```
unmaned_store_shoplifting_detection/
├── .env                    # 환경 변수 (S3 키 등)
├── requirements.txt        # Python 패키지
├── Dockerfile             # Docker 이미지 빌드
├── docker-compose.yml     # 서비스 오케스트레이션
│
├── ai_model/              # AI 모델 및 학습 코드
│   ├── args.py            # 전체 설정
│   ├── checkpoints/       # ★ 학습된 모델 가중치 (.pth)
│   ├── models/            # 모델 구조
│   ├── datasets/          # 데이터셋 로더
│   └── extract_skeleton/  # YOLO skeleton 추출
│
├── ai_server/             # AI 추론 서버
│   ├── celery_app.py      # Celery 설정
│   ├── tasks.py           # Celery 작업 정의
│   └── inference.py       # 실제 추론 로직
│
└── backend/               # 웹 서버
    ├── main.py            # FastAPI 서버
    ├── database.py        # SQLite DB
    └── templates/         # HTML UI
```

---

## 사전 준비

### 1. 필수 소프트웨어 설치
- **Docker Desktop** (Windows/Mac) 또는 Docker Engine (Linux)
- **NVIDIA Docker** (GPU 사용 시)
  ```bash
  # NVIDIA Container Toolkit 설치 (Linux)
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

### 2. 환경 변수 설정 (.env 파일)
프로젝트 루트에 `.env` 파일 생성 후 S3 정보 입력:

```env
S3_BUCKET_NAME=your-bucket-name
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
S3_REGION=ap-northeast-2
```

### 3. 모델 가중치 확인
`ai_model/checkpoints/` 폴더에 다음 파일들이 있어야 합니다:
- `stgnf_head_fin.pth`
- `stgnf_arms_fin.pth`
- `stgnf_body_fin.pth`
- `stgnf_legs_fin.pth`
- `stgnf_all_fin.pth`
- `attention_fin.pth` ← **가장 중요**

---

## 실행 방법

### 1. Docker 이미지 빌드 및 실행
프로젝트 루트 폴더에서:

```bash
# 한 줄로 실행 (빌드 + 시작)
docker-compose up --build

# 백그라운드 실행 (로그 안 보임)
docker-compose up -d --build
```

### 2. 서비스 상태 확인
```bash
# 컨테이너 상태 확인
docker-compose ps

# AI Worker 로그 확인 (모델 로딩 완료 확인)
docker-compose logs -f ai-worker

# 다음 메시지가 나오면 준비 완료:
# [AI Server] Detector 준비 완료!
```

### 3. 웹 접속
브라우저에서 `http://localhost:8000` 접속
- 영상 업로드
- 분석 진행 (로딩 화면)
- 결과 확인 (도난 구간, 시작 시간)

---

## 개발/테스트 모드

### 로컬에서 개별 실행 (Docker 없이)

#### 1. Redis 시작
```bash
# Windows (WSL 또는 Redis 설치 필요)
redis-server

# Mac (Homebrew)
brew services start redis
```

#### 2. Backend 실행
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. AI Worker 실행 (별도 터미널)
```bash
# 가상환경 활성화 후
celery -A ai_server.celery_app worker --loglevel=info --pool=solo
```

---

## 주요 설정

### 모델 설정 (ai_model/args.py)
- `SEG_LEN = 24`: 시퀀스 길이 (24프레임)
- `TRAIN_STRIDE = 12`: 학습 stride
- `EVAL_STRIDE = 1`: 평가 stride (모든 프레임)
- `VID_RES = [1920, 1080]`: 비디오 해상도
- `NORMALIZE = True`: skeleton 정규화

### 추론 설정 (ai_server/inference.py)
- `min_frames=12`: 최소 연속 프레임 수 (12프레임 이상 사람 있어야 탐지)
- `threshold=0.5`: 도난 판별 임계값

---

## 문제 해결

### 1. 모델 로딩 실패
```
FileNotFoundError: Attention 체크포인트 없음
```
→ `ai_model/checkpoints/attention_fin.pth` 파일이 있는지 확인

### 2. GPU 인식 안 됨
```
[AI Server] 모델 로딩 시작... (Device: cpu)
```
→ NVIDIA Docker 설치 확인, `docker-compose.yml`에 GPU 설정 확인

### 3. Redis 연결 오류
```
celery.exceptions.OperationalError
```
→ Redis 컨테이너가 먼저 시작되었는지 확인 (`docker-compose ps`)

### 4. Import 오류
```
ModuleNotFoundError: No module named 'ai_model'
```
→ Dockerfile에서 `ENV PYTHONPATH=/app` 설정 확인

---

## 배포 체크리스트

### 운영 배포 전 확인사항
- [ ] `.env` 파일에 실제 S3 키 입력
- [ ] 모든 모델 가중치(.pth) 파일 존재 확인
- [ ] `attention_fin.pth` F1 score 확인 (최소 90% 이상 권장)
- [ ] GPU 사용 가능 환경 (NVIDIA Docker)
- [ ] 충분한 메모리 (최소 8GB RAM, 6GB VRAM)
- [ ] 테스트 영상으로 전체 파이프라인 검증

### 성능 모니터링
```bash
# GPU 사용률 확인
nvidia-smi

# 컨테이너 리소스 확인
docker stats

# AI Worker 로그 실시간 확인
docker-compose logs -f ai-worker
```

---

## 종료 및 재시작

```bash
# 서비스 종료
docker-compose down

# 데이터 보존하며 종료
docker-compose stop

# 재시작
docker-compose start

# 완전 삭제 (이미지까지)
docker-compose down --rmi all
```

---
