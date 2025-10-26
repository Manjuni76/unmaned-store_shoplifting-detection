# YOLO Skeleton Extractor 설치 및 실행 가이드

## 필요 패키지 설치

```bash
pip install ultralytics opencv-python torch torchvision
```

## 사용법

### 1. 단일 비디오에서 스켈레톤 추출

```bash
python yolo_skeleton_extractor.py --mode single --input "video_path.mp4" --output "output.json"
```

### 2. 분할된 데이터에서 배치 스켈레톤 추출

```bash
python yolo_skeleton_extractor.py --mode batch --input "data_splits_balanced/train_feature/feature_extraction_info.json" --output "skeleton_data/train_feature"
```

### 3. 전체 분할 데이터 스켈레톤 추출 (권장)

```bash
python run_skeleton_extraction.py --mode extract_all
```

## 특징

1. **YOLO Pose 사용**: 최신 YOLOv8/v11-pose 모델로 빠르고 정확한 포즈 추출
2. **COCO17 포맷**: 17개 키포인트 (논문과 동일)
3. **신뢰도 필터링**: 낮은 신뢰도 키포인트 자동 제거
4. **배치 처리**: 분할된 데이터 자동 처리
5. **progress 표시**: 실시간 진행 상황 확인

## 출력 형식

```json
{
  "person_1": {
    "0": {
      "keypoints": [
        [x, y, confidence],  // nose
        [x, y, confidence],  // left_eye
        ...                  // 총 17개 키포인트
      ]
    },
    "1": { ... }  // 다음 프레임
  },
  "person_2": { ... }
}
```

## 논문 방식과의 차이점

- **YOLOv8**: 기존 YOLOv8 + HRNet 대신 YOLOv8-pose 통합 모델 사용
- **ByteTrack**: 현재는 간단한 per-frame 탐지 (추후 추가 가능)
- **전처리**: data_filter.py에서 별도 처리 예정

## 다음 단계

1. 스켈레톤 추출 완료
2. data_filter.py로 신뢰도 필터링, 선형보간, 스무딩 적용
3. 특징추출 모델 학습 시작