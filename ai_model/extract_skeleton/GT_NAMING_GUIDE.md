# GT 파일 및 의미있는 네이밍 시스템 가이드

## 개요
`dataset_processor.py`가 추가로 개선되어 다음과 같은 기능이 추가되었습니다:
- **GT(Ground Truth) 파일**: `labels.npy` → `ground_truth.npy` (테스트 세트만)
- **의미있는 비디오 네이밍**: `1, 2, 3...` → `normal_shopping_001_원본파일명`
- **비디오 메타데이터**: 각 비디오의 상세 정보 JSON 파일

## 새로운 데이터 구조

### 폴더 구조
```
dataset_output/
├── train/
│   ├── pose_data.json          # 포즈 데이터 (PoseLift 형식)
│   └── video_metadata.json     # 비디오 메타데이터
└── test/
    ├── pose_data.json          # 포즈 데이터 (PoseLift 형식)
    ├── ground_truth.npy        # GT 레이블 (테스트용만)
    └── video_metadata.json     # 비디오 메타데이터
```

### 변경사항 요약
| 항목 | 기존 | 개선 후 |
|------|------|---------|
| NPY 파일 | `train/labels.npy`, `test/labels.npy` | `test/ground_truth.npy` (테스트만) |
| 비디오 ID | `1_person_0`, `2_person_1` | `normal_shopping_001_video123_person_0` |
| 메타데이터 | 없음 | `video_metadata.json` |

## 의미있는 네이밍 시스템

### 비디오 네이밍 규칙
```
{카테고리}_{행동유형}_{순번:3자리}_{원본파일명}
```

#### 예시
- **정상 행동**: `normal_shopping_001_video123`
- **이상 행동**: `abnormal_shoplifting_002_shoplifting_scene_456`
- **Person ID**: `normal_shopping_001_video123_person_0`

### 카테고리 분류
- `normal_shopping`: 정상 쇼핑 행동
- `abnormal_shoplifting`: 이상 행동 (절도)

## 메타데이터 구조

### video_metadata.json 예시
```json
{
  "normal_shopping_001_video123": {
    "original_path": "D:\\AI-HUB_shoping\\shoping_data\\Training\\video123.mp4",
    "category": "normal",
    "frame_count": 300,
    "person_count": 2
  },
  "abnormal_shoplifting_002_scene456": {
    "original_path": "D:\\AI-HUB_shoplifting\\shoplift_data\\Training\\scene456.mp4",
    "category": "abnormal", 
    "frame_count": 150,
    "person_count": 1
  }
}
```

### 메타데이터 필드 설명
- `original_path`: 원본 비디오 파일 경로
- `category`: 카테고리 (`normal` 또는 `abnormal`)
- `frame_count`: 총 프레임 수
- `person_count`: 검출된 사람 수

## GT(Ground Truth) 파일

### 변경 사항
- **기존**: `train/labels.npy`, `test/labels.npy`
- **개선**: `test/ground_truth.npy` (테스트 세트만)

### 이유
- 훈련 데이터는 라벨이 필요하지 않음 (비지도 학습)
- 테스트 시에만 Ground Truth가 필요
- 파일명이 용도를 명확히 표현

## 사용 예시

### Python에서 데이터 로드
```python
import json
import numpy as np

# 메타데이터 로드
with open('dataset_output/test/video_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# GT 라벨 로드
ground_truth = np.load('dataset_output/test/ground_truth.npy')

# 포즈 데이터 로드
with open('dataset_output/test/pose_data.json', 'r') as f:
    pose_data = json.load(f)

# 특정 비디오 정보 확인
for video_name, info in metadata.items():
    print(f"비디오: {video_name}")
    print(f"  카테고리: {info['category']}")
    print(f"  프레임 수: {info['frame_count']}")
    print(f"  사람 수: {info['person_count']}")
```

### 비디오별 분석
```python
# 정상/이상 비디오 분리
normal_videos = {k: v for k, v in metadata.items() if v['category'] == 'normal'}
abnormal_videos = {k: v for k, v in metadata.items() if v['category'] == 'abnormal'}

print(f"정상 비디오: {len(normal_videos)}개")
print(f"이상 비디오: {len(abnormal_videos)}개")
```

## 장점

### 1. 명확한 파일 구조
- GT 파일은 테스트에만 필요하므로 테스트 세트에만 생성
- 파일명이 용도를 명확히 나타냄

### 2. 의미있는 네이밍
- 비디오 카테고리와 순번을 이름에 포함
- 원본 파일명 보존으로 추적 가능

### 3. 풍부한 메타데이터
- 각 비디오의 상세 정보 제공
- 데이터 분석 및 디버깅에 유용

### 4. 확장성
- 새로운 카테고리 쉽게 추가 가능
- 메타데이터 필드 확장 가능

## 실행 결과 예시
```
설정된 비율:
  - 정상:이상 = 70.0%:30.0%
  - Train:Test = 4:1

[1/426] normal_shopping_001_video123 - 성공
[2/426] normal_shopping_002_scene456 - 성공
[3/426] abnormal_shoplifting_001_theft_action - 성공

저장 완료: train/ - pose_data.json, video_metadata.json
저장 완료: test/ - pose_data.json, ground_truth.npy, video_metadata.json
```