# Unmanned Store Shoplifting Detection

무인 상점 절도 탐지 시스템 - STG-NF(Spatio-Temporal Graph Normalizing Flow) 기반

## 📌 프로젝트 개요

이 프로젝트는 스켈레톤 기반 행동 인식을 활용하여 무인 상점에서의 절도 행위를 자동으로 탐지하는 AI 시스템입니다.

### 주요 특징

- **STG-NF 모델**: 정상 행동 패턴을 학습하여 이상 행동 탐지
- **부위별 학습**: 팔, 다리, 몸통 등 신체 부위별 독립적 학습 가능
- **2단계 학습 파이프라인**: 
  1. STG-NF로 정상 패턴 학습
  2. MLP 분류기로 정상/이상 분류

## 🗂️ 프로젝트 구조

```
.
├── ai_model/                      # AI 모델 및 학습 코드
│   ├── model/                     # 모델 구현
│   │   ├── stg_nf.py             # STG-NF 모델
│   │   └── README.md             # 모델 문서
│   ├── data_preprocessing/        # 데이터 전처리
│   │   ├── data_preprocess.py    # 키포인트 전처리 파이프라인
│   │   ├── utils_data_loader.py  # 데이터 로더 유틸리티
│   │   └── main_preprocessing.py # 전처리 메인 스크립트
│   ├── extract_skeleton/          # 스켈레톤 추출
│   ├── train_pipeline.py          # 통합 학습 파이프라인
│   ├── test_model.py             # 모델 검증 스크립트
│   ├── README.md                 # AI 모델 상세 문서
│   └── USAGE.md                  # 사용 가이드
├── data_split/                    # 데이터 분할 스크립트
│   └── data_split.py             # 데이터셋 분할 및 GT 생성
└── README.md                     # 이 파일

```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Conda 환경 생성
conda env create -f ai_model/environment/environment.yml
conda activate unmaned_shoplifting

# 또는 pip으로 설치
pip install torch torchvision numpy scikit-learn tqdm opencv-python
```

### 2. 데이터 준비

데이터를 준비하고 분할합니다:

```bash
cd data_split
python data_split.py
```

### 3. 모델 검증 (선택사항)

학습 전에 모델이 정상적으로 작동하는지 확인:

```bash
cd ai_model
python test_model.py
```

### 4. 학습 실행

```bash
cd ai_model
python train_pipeline.py
```

## 📖 상세 문서

- **AI 모델 상세 정보**: [ai_model/README.md](ai_model/README.md)
- **사용 가이드**: [ai_model/USAGE.md](ai_model/USAGE.md)
- **모델 구현**: [ai_model/model/README.md](ai_model/model/README.md)

## 🎯 학습 파이프라인

### Stage 1: STG-NF 정상 데이터 학습
- 정상 행동 패턴만을 사용하여 STG-NF 모델 학습
- 정상 행동의 분포를 학습
- 출력: `checkpoints/stgnf_[부위].pth`

### Stage 2: MLP 분류기 학습
- STG-NF를 freeze하고 MLP만 학습
- 정상 + 이상 데이터로 분류 학습
- 출력: `checkpoints/full_model_[부위].pth`

### Stage 3: 평가
- 테스트 데이터로 성능 평가
- Accuracy, AUC-ROC, AUC-PR 메트릭 계산

## 🔧 주요 설정

### 부위별 학습

```python
# train_pipeline.py에서 설정
JOINT_SUBSET_MAP = {
    'arms': [2,3,4,5,6,7],          # 팔
    'legs': [8,9,10,11,12,13],      # 다리
    'body': [1,2,5,8,11],           # 몸통
    'head': [0,14,15,16,17],        # 머리
    'all': None                      # 전체
}
```

### 하이퍼파라미터

```python
args = {
    'seg_len': 12,              # 시퀀스 길이
    'seg_stride': 6,            # Stride
    'batch_size': 32,           # 배치 크기
    'epochs_stgnf': 50,         # STG-NF 에폭
    'epochs_mlp': 30,           # MLP 에폭
    'lr_stgnf': 1e-4,          # STG-NF 학습률
    'lr_mlp': 1e-3,            # MLP 학습률
}
```

## 📊 예상 결과

```
[RESULTS]
Accuracy: 87.34%
AUC-ROC: 0.9234
AUC-PR: 0.8976
```

## 🐛 문제 해결

### 일반적인 문제

1. **CUDA out of memory**
   ```python
   args['batch_size'] = 16  # 배치 크기 감소
   ```

2. **데이터 파일 없음**
   - `data_split.py`를 먼저 실행
   - 경로가 올바른지 확인

3. **모듈 import 오류**
   - 의존성 설치 확인
   - Python 3.11 사용 확인

자세한 문제 해결은 [USAGE.md](ai_model/USAGE.md) 참조

## 📝 개발 로그

### 최근 업데이트

- **2025-11-07**: 
  - STG-NF 모델 구현 완료
  - 학습 파이프라인 import 경로 수정
  - 모델 검증 스크립트 추가
  - 문서 업데이트

## 🔬 기술 스택

- **Deep Learning**: PyTorch
- **Pose Estimation**: OpenPose (COCO-18 keypoints)
- **Model**: STG-NF (Spatio-Temporal Graph Normalizing Flow)
- **Data Processing**: NumPy, OpenCV

## 📚 참고 자료

- STG-NF 논문: [Normalizing Flows for Human Pose Anomaly Detection (ICCV 2023)](https://arxiv.org/abs/2211.10946)
- 데이터셋: AI-HUB 쇼핑몰 행동 데이터셋

## 👥 기여

기여를 환영합니다! 이슈나 PR을 자유롭게 제출해주세요.

## 📄 라이센스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.
