# 무인매장 도난 탐지 시스템 - Docker 배포 가이드

## 프로젝트 개요
STG-NF + Attention 기반 도난 탐지 AI 시스템

### 주요 기능
1. 영상 업로드 (MP4 형식)
2. YOLO11 기반 skeleton 추출
3. 12프레임 이상 사람이 있는 구간만 도난 탐지 진행
4. STG-NF + Attention 모델로 4단계 위험도 평가
5. 도난 의심 구간 표시 및 결과 반환

### 배포 특징
- **CPU 버전**: GPU 없이도 실행 가능, RTX 5080 cuda 버전 오류로 CPU 버전으로 배포 진행
- **처리 속도**: 1분 영상 약 40초 소요 (Ryzen 9 9950X 기준)
- **정확도**: GPU 버전과 동일 (AUC-ROC 97.3%)

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
│   ├── checkpoints/       # **학습된 모델 가중치 (.pth)**
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

### 1. 시스템 요구사항
- **OS**: Windows 10/11, macOS, Linux
- **RAM**: 8GB 이상 권장 (16GB 권장)
- **저장 공간**: 10GB 이상
- **CPU**: 멀티코어 프로세서 권장

> **참고**: 이 프로젝트는 CPU 버전으로 구성되어 있어 GPU 없이도 실행 가능합니다.
> 처리 속도는 느리지만 정확도는 동일합니다.

### 2. 필수 소프트웨어 설치
- **Docker Desktop** (Windows/Mac) 또는 Docker Engine (Linux)
- 설치 후 재부팅 필요
- Docker Desktop이 실행 중인지 확인

### 3. AWS S3 설정 (필수)

#### S3 버킷 생성
1. [AWS Console](https://console.aws.amazon.com/s3/) 접속
2. "버킷 만들기" 클릭
3. 버킷 이름 입력 (예: `shoplifting-detection-videos`)
4. 리전 선택 (예: `ap-northeast-2` - 서울)
5. 기본 설정으로 버킷 생성

#### IAM 사용자 생성 및 액세스 키 발급
1. [IAM Console](https://console.aws.amazon.com/iam/) 접속
2. "사용자" → "사용자 추가" 클릭
3. 사용자 이름 입력 (예: `shoplifting-app`)
4. "액세스 키 - 프로그래밍 방식 액세스" 선택
5. 권한 설정:
   - "기존 정책 직접 연결" 선택
   - `AmazonS3FullAccess` 정책 선택
6. 액세스 키 ID와 비밀 액세스 키 저장 (한 번만 표시됨)

#### 환경 변수 설정 (.env 파일)
프로젝트 루트에 `.env` 파일 생성:

```env
S3_BUCKET_NAME=shoplifting-detection-videos
S3_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
S3_SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
S3_REGION=ap-northeast-2
```

> **중요**: 
> - S3는 **필수**입니다. 영상 업로드 및 결과 저장에 사용됩니다.
> - AWS 프리 티어로도 충분히 사용 가능합니다 (월 5GB 저장 무료)

### 4. 모델 가중치 확인
`ai_model/checkpoints/` 폴더에 다음 모델 가중치 파일들이 있어야 합니다:
- `stgnf_head.pth` - 머리 부위 모델
- `stgnf_arms.pth` - 팔 부위 모델
- `stgnf_body.pth` - 몸통 부위 모델
- `stgnf_legs.pth` - 다리 부위 모델
- `stgnf_all.pth` - 전체 모델
- `attention_classifier.pth` - **최종 분류기 (필수)**


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
브라우저에서 `http://localhost:8001` 접속

**사용 방법:**
1. 영상 업로드 (MP4 파일)
2. 분석 시작 버튼 클릭
3. 로딩 화면 대기 (1분 영상 약 40초 소요)
4. 결과 확인:
   - 위험도 레벨 (정상/주의/의심/확실)
   - 도난 의심 구간 (시작~종료 시간)
   - 영상 플레이어 (구간 클릭 시 자동 이동)

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

#### 기본 설정
- `min_frames=12`: 최소 연속 프레임 수 (12프레임 = 약 0.4초)
- `fps=30`: 프레임 레이트 (1초 당 30프레임)

#### 4단계 위험도 평가 기준
| 위험도 | 점수 범위 | 색상 | 의미 | 조치 |
|--------|------------|------|------|------|
| **정상** | < 0.80 | 회색 | 도난 행동 없음 | 필요 없음 |
| **주의 필요** | 0.80 ~ 0.87 | 노란색 | 의심스러운 동작 감지 | 영상 확인 권장 |
| **도난 의심** | 0.87 ~ 0.98 | 주황색 | 도난 가능성 높음 | 관리자 확인 필수 |
| **확실한 도난** | ≥ 0.98 | 빨간색 | 도난 행동 확실 | 즉시 대응 필요 |

> **참고**: Threshold를 0.98로 설정하여 오탐지(False Positive)를 최소화했습니다.



#### 처리 속도
- **Ryzen 9 9950X**: 1분 영상 약 40초

#### 정확도
- **AUC-ROC**: 97.3% (GPU 버전과 동일)
- **AUC-PR**: 70.2%
- **EER**: 7.3%

#### 리소스 사용량
- **RAM**: 약 4-6GB
- **CPU**: 80-100% (분석 중)
- **디스크**: 10GB 이상 권장

> **참고**: PyTorch CPU 버전 사용으로 GPU 없이도 실행 가능합니다.

---

## 문제 해결

### 1. Docker 관련 문제

#### Docker Desktop이 실행되지 않음
```
error during connect: open //./pipe/dockerDesktopLinuxEngine
```
**해결방법:**
- Docker Desktop 실행 (작업 표시줄 아이콘 확인)
- 30초~1분 대기 후 재시도

#### 포트 충돌
```
Bind for 0.0.0.0:8001 failed: port is already allocated
```
**해결방법:**
```bash
# 사용 중인 프로세스 확인
netstat -ano | findstr :8001

# 또는 docker-compose.yml에서 포트 변경
ports:
  - "8002:8000"  # 8001 → 8002로 변경
```

### 2. 모델 로딩 문제

#### 모델 가중치 파일 없음
```
FileNotFoundError: Attention 체크포인트 없음
```
**해결방법:**
- `ai_model/checkpoints/` 폴더 확인
- 필수 파일: `attention_classifier.pth`
- 모든 STG-NF 파일 (5개) 확인

#### CUDA/GPU 관련 오류 (무시 가능)
```
[AI Server] 모델 로딩 시작... (Device: cpu)
```
**해결방법:**
- 정상 동작 (CPU 버전이므로 문제 없음)
- GPU 사용을 원하면 `requirements_full.txt`로 로컬 실행

### 3. 분석 실패

#### 유효 구간 없음
```
유효한 구간이 12프레임 미만입니다
```
**원인:** 영상에 사람이 충분히 나오지 않음
**해결방법:**
- 최소 12프레임(8초) 이상 사람이 명확히 보이는 영상 사용
- 카메라 각도가 너무 멀거나 가려지지 않도록



### 4. Redis 연결 오류
```
celery.exceptions.OperationalError
```
**해결방법:**
```bash
# Redis 컨테이너 상태 확인
docker-compose ps redis

# Redis 재시작
docker-compose restart redis
```

### 5. Import 오류
```
ModuleNotFoundError: No module named 'ai_model'
```
**해결방법:**
- Dockerfile에서 `ENV PYTHONPATH=/app` 설정 확인
- Docker 재빌드: `docker-compose up --build`

---

## 배포 체크리스트

### 운영 배포 전 확인사항
- [ ] Docker Desktop 설치 및 실행 상태 확인
- [ ] **AWS S3 버킷 생성 완료 (필수)**
- [ ] **IAM 사용자 액세스 키 발급 완료 (필수)**
- [ ] **`.env` 파일 설정 완료 (필수)**
  - S3_BUCKET_NAME
  - S3_ACCESS_KEY
  - S3_SECRET_KEY
  - S3_REGION
- [ ] 모든 모델 가중치(.pth) 파일 6개 확인
  - STG-NF: `stgnf_head.pth`, `stgnf_arms.pth`, `stgnf_body.pth`, `stgnf_legs.pth`, `stgnf_all.pth`
  - Attention: `attention_classifier.pth`
- [ ] 충분한 메모리 (8GB 이상 RAM)
- [ ] 충분한 저장 공간 (10GB 이상)
- [ ] 테스트 영상으로 전체 파이프라인 검증

### 성능 모니터링
```bash
# 컨테이너 리소스 확인
docker stats

# AI Worker 로그 실시간 확인
docker-compose logs -f ai-worker

# Backend 로그 확인
docker-compose logs -f backend
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
