# 무인 상점 절도 탐지 AI 모델 학습 파이프라인

## 📌 개요

이 파이프라인은 다음과 같은 3단계로 구성됩니다:

1. **STG-NF 정상 데이터 학습**: 정상 행동 패턴만 학습 (train_data.json - 783개)
2. **MLP 분류기 학습**: STG-NF를 freeze하고 MLP만 학습 (mlp_train_data.json - 정상 356 + 이상 356)
3. **테스트 평가**: test_data.json으로 평가 (정상 427 + 이상 285)

## 🗂️ 디렉토리 구조

```
ai_model/
├── train_pipeline.py          # 통합 학습 파이프라인
├── data_preprocessing/
│   ├── data_preprocess.py
│   ├── data_split.py
│   └── ...
├── extract_skeleton/          # 스켈레톤 추출 (이미 완료)
└── model/                     # 모델 정의

data/
├── train_data_skeleton_data/  # 정상 데이터 스켈레톤 (783개)
├── mlp_train_data_skeleton_data/  # MLP 학습용 스켈레톤 (712개)
├── test_data_skeleton_data/   # 테스트 스켈레톤 (712개)
└── gt/                        # Ground Truth 라벨

data_split/
└── output/
    ├── train_data.json        # 정상 783개
    ├── mlp_train_data.json    # 정상 356 + 이상 356
    ├── test_data.json         # 정상 427 + 이상 285
    └── GT/

STG-NF_AI-HUB/                 # 기존 STG-NF 코드베이스
├── models/
│   └── STG_NF/
├── dataset.py                 # 관절 부위 매핑
├── train_eval.py
└── ...

checkpoints/                   # 저장된 모델 체크포인트
├── stgnf_arms.pth
└── full_model_arms.pth
```

## 🚀 사용 방법

### 1. 환경 설정

```bash
conda activate unmaned_shoplifting
cd ai_model
```

### 2. 학습 실행

```python
python train_pipeline.py
```

### 3. 설정 변경

`train_pipeline.py`의 `main()` 함수에서 `args` 딕셔너리 수정:

```python
args = {
    # 경로 설정 (기본값 그대로 사용)
    'train_json': '../data_split/output/train_data.json',
    'mlp_train_json': '../data_split/output/mlp_train_data.json',
    'test_json': '../data_split/output/test_data.json',
    
    # 스켈레톤 데이터 경로
    'train_skeleton_path': '../data/train_data_skeleton_data',
    'mlp_skeleton_path': '../data/mlp_train_data_skeleton_data',
    'test_skeleton_path': '../data/test_data_skeleton_data',
    
    # 체크포인트 경로
    'stgnf_checkpoint': './checkpoints/stgnf_arms.pth',
    'full_model_checkpoint': './checkpoints/full_model_arms.pth',
    
    # 모델 설정
    'seg_len': 12,              # 시퀀스 길이
    'seg_stride': 6,            # 시퀀스 stride
    'joint_subset': JOINT_SUBSET_MAP['arms'],  # 사용할 관절 부위
    
    # 학습 설정
    'batch_size': 32,
    'epochs_stgnf': 50,         # STG-NF 학습 에폭
    'epochs_mlp': 30,           # MLP 학습 에폭
    'lr_stgnf': 1e-4,
    'lr_mlp': 1e-3,
    
    # 기타
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42
}
```

## 🎯 관절 부위 선택

`joint_subset` 설정으로 학습할 관절 부위를 선택할 수 있습니다:

```python
from dataset import JOINT_SUBSET_MAP

# 사용 가능한 옵션:
JOINT_SUBSET_MAP['arms']         # [2,3,4,5,6,7] 전체 팔
JOINT_SUBSET_MAP['legs']         # [8,9,10,11,12,13] 전체 다리
JOINT_SUBSET_MAP['left_arm']     # [5,6,7] 왼쪽팔
JOINT_SUBSET_MAP['right_arm']    # [2,3,4] 오른쪽팔
JOINT_SUBSET_MAP['left_leg']     # [11,12,13] 왼쪽다리
JOINT_SUBSET_MAP['right_leg']    # [8,9,10] 오른쪽다리
JOINT_SUBSET_MAP['body']         # [1,2,5,8,11] 몸통
JOINT_SUBSET_MAP['head']         # [0,14,15,16,17] 머리
JOINT_SUBSET_MAP['arm+body']     # 상체 전체
JOINT_SUBSET_MAP['head+body']    # 머리+몸통
JOINT_SUBSET_MAP['all']          # None (모든 관절)
```

## 📊 학습 과정

### Step 1: STG-NF 정상 데이터 학습
- **데이터**: train_data.json (정상 783개)
- **목적**: 정상 행동 패턴 학습
- **출력**: `checkpoints/stgnf_arms.pth`

### Step 2: MLP 분류기 학습
- **데이터**: mlp_train_data.json (정상 356 + 이상 356)
- **목적**: STG-NF 특징으로 정상/이상 분류
- **STG-NF**: Freeze (학습하지 않음)
- **MLP**: 학습
- **출력**: `checkpoints/full_model_arms.pth`

### Step 3: 테스트 평가
- **데이터**: test_data.json (정상 427 + 이상 285)
- **메트릭**:
  - Accuracy
  - AUC-ROC
  - AUC-PR

## 🔧 고급 사용법

### 부위별 학습 실행

```python
# 팔만 학습
args['joint_subset'] = JOINT_SUBSET_MAP['arms']
args['stgnf_checkpoint'] = './checkpoints/stgnf_arms.pth'
args['full_model_checkpoint'] = './checkpoints/full_model_arms.pth'

# 다리만 학습
args['joint_subset'] = JOINT_SUBSET_MAP['legs']
args['stgnf_checkpoint'] = './checkpoints/stgnf_legs.pth'
args['full_model_checkpoint'] = './checkpoints/full_model_legs.pth'
```

### 체크포인트에서 이어서 학습

```python
# STG-NF 모델 로드
stg_nf_model = STG_NF(**model_args).to(device)
stg_nf_model.load_state_dict(torch.load('./checkpoints/stgnf_arms.pth'))

# MLP 학습부터 시작
full_model = train_mlp_classifier(stg_nf_model, args)
```

## 📈 결과 확인

학습이 완료되면 다음과 같은 결과가 출력됩니다:

```
[RESULTS]
Accuracy: 87.34%
AUC-ROC: 0.9234
AUC-PR: 0.8976
```

## ⚠️ 주의사항

1. **메모리**: 배치 크기를 GPU 메모리에 맞게 조정하세요
2. **시드**: 재현성을 위해 시드를 고정했습니다
3. **데이터**: 스켈레톤 JSON 파일이 정확한 경로에 있어야 합니다
4. **관절 인덱스**: COCO-18 포맷 기준입니다

## 🐛 문제 해결

### 1. 메모리 부족
```python
args['batch_size'] = 16  # 또는 8
```

### 2. 스켈레톤 파일 없음
- `data/` 폴더에 스켈레톤 JSON 파일들이 있는지 확인
- `extract_skeleton/` 폴더의 스크립트로 먼저 추출

### 3. 모델 로딩 오류
- STG-NF 모델 파라미터가 올바른지 확인
- 체크포인트 파일 경로 확인

## 📝 TODO

- [ ] 앙상블 모델 (부위별 모델 결합)
- [ ] 프레임 단위 이상 구간 탐지
- [ ] 시각화 도구
- [ ] 하이퍼파라미터 자동 튜닝

## 📚 참고 자료

- STG-NF 논문: [Normalizing Flows for Human Pose Anomaly Detection (ICCV 2023)](https://arxiv.org/abs/2211.10946)
- 데이터셋: AI-HUB 쇼핑몰 행동 데이터셋
