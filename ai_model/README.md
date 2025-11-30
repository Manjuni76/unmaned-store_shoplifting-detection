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

### 1. 환경 준비
- Python 3.8+ 및 필요한 패키지 설치
- (권장) conda 환경 사용: `conda env create -f environment/environment.yml`
- 환경 활성화: `conda activate <env_name>`

### 2. 데이터 준비
- `data/` 폴더에 skeleton 데이터 및 GT 파일 배치
  - 예시: `data/train_stgnf/skeleton_data/`, `data/gt/test_gt/`

### 3. 모델 학습 설정
- 모든 하이퍼파라미터 및 경로는 `ai_model/args.py`에서 관리
  - 시퀀스 길이, stride, 배치 크기, epoch 등 직접 수정 가능
  - 커맨드라인 인자도 지원 (예: `--epochs 100`)

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
- skeleton 데이터는 `extract_skeleton/skeleton_extract.py`로 생성
- 모델 학습/평가 시 GPU 사용 권장 (`args.py`에서 `DEVICE` 설정)
- 각 스크립트 실행 전 환경 및 경로 설정을 반드시 확인하세요.

---

## 문의/기여
- 코드/설정 관련 문의는 이 README 또는 args.py 주석 참고
- 추가 개선/기여는 Pull Request로 환영합니다.
