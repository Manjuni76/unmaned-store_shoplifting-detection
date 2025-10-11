# 데이터 분할 스크립트 실행 가이드

## 요구사항
- Python 3.7+
- OpenCV
- NumPy

## 설치 방법
```bash
pip install opencv-python numpy
```

## 사용 방법
```bash
cd data_split
python data_split.py
```

## 출력 구조
```
output/
├── train/              # 정상 데이터만 (mlp_train + test와 같은 개수)
├── mlp_train/          # 정상:이상 = 5:5
├── test/               # 정상:이상 = 6:4
└── gt/                 # Ground Truth 파일들
    ├── mlp_train_gt/   # mlp_train용 GT (.npy 파일)
    └── test_gt/        # test용 GT (.npy 파일)
```

## GT 파일 형식
- .npy 파일로 저장
- 0: 정상 프레임
- 1: 이상 프레임 (theft_start ~ theft_end)
- 배열 크기: 비디오의 총 프레임 수와 동일

## 데이터 분할 조건
1. 모든 이상 데이터 사용
2. mlp_train: 정상 5 : 이상 5
3. test: 정상 6 : 이상 4
4. train: 정상 데이터만 (mlp_train + test만큼의 개수)
5. XML에서 theft_start, theft_end 구간 자동 추출