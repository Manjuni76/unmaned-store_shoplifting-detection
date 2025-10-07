#데이터 슬라이딩 윈도우 스퀀스 분할
# 데이터 COCO18 형태 변환
# 데이터 증강 (슬라이딩 윈도우 시퀀스 분할)
# 데이터 정규화 및 표준화

import numpy as np
from typing import Dict, Any, List, Tuple

# 실제 JSON 데이터 구조 (리스트 형태):
# {
#     "person_1": {
#         "0": {  # frame_number
#             "keypoints": [
#                 [100, 150, 0.8],  # 관절점 0 (코) - [x, y, confidence]
#                 [90, 140, 0.7],   # 관절점 1 (왼쪽 눈) - [x, y, confidence]
#                 [95, 145, 0.6],   # 관절점 2 (오른쪽 눈) - [x, y, confidence]
#                 ...               # 총 17개 관절점 (COCO17 형식)
#             ]
#         },
#         "1": { ... }  # 다음 프레임
#     },
#     "person_2": { ... }
# }

def sliding_window_sequence_split(data, window_size = 24, stride = 4):
    """
    슬라이딩 윈도우를 사용하여 시퀀스를 분할합니다.
    전체 skeleton 스코어를 일정한 길이의 세그먼트로 분할하여 학습 데이터를 증강합니다.
    
    Args:
        data: 전처리된 키포인트 데이터
        window_size: 윈도우 크기 (프레임 수, 기본값: 30)
        stride: 슬라이딩 스텝 크기 (기본값: 15)
        
    Returns:
        분할된 시퀀스 리스트 [{"person_id": str, "sequence": np.array, "frames": list}]
    """
    sequences = []
    
    for person_id, frames in data.items():
        # 프레임을 번호 순서로 정렬
        sorted_frame_nums = sorted(frames.keys(), key=int)
        
        # 프레임이 window_size보다 작으면 건너뛰기
        if len(sorted_frame_nums) < window_size:
            continue
            
        # 슬라이딩 윈도우로 시퀀스 분할
        for start_idx in range(0, len(sorted_frame_nums) - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            # 윈도우에 해당하는 프레임들 추출
            window_frames = sorted_frame_nums[start_idx:end_idx]
            
            # 키포인트 시퀀스 생성 (window_size, num_keypoints, 3)
            sequence_data = []
            valid_sequence = True
            
            for frame_num in window_frames:
                keypoints = frames[frame_num]["keypoints"]
                
                # 모든 키포인트가 무효한 프레임이 있으면 해당 시퀀스는 제외
                valid_keypoints = [kp for kp in keypoints if kp[2] > 0]  # confidence > 0
                if len(valid_keypoints) < len(keypoints) * 0.5:  # 절반 이상이 무효하면
                    valid_sequence = False
                    break
                    
                sequence_data.append(keypoints)
            
            # 유효한 시퀀스만 추가
            if valid_sequence:
                sequence_array = np.array(sequence_data)  # (window_size, num_keypoints, 3)
                
                sequences.append({
                    "person_id": person_id,
                    "sequence": sequence_array,
                    "frames": window_frames,
                    "start_frame": int(window_frames[0]),
                    "end_frame": int(window_frames[-1])
                })
    
    return sequences


def convert_to_coco18(data):
    """
    COCO17 형태를 COCO18 형태로 변환합니다.
    목 키포인트를 추가하여 18개 키포인트로 확장합니다.
    
    COCO17: 0-코, 1-왼눈, 2-오른눈, 3-왼귀, 4-오른귀, 5-왼어깨, 6-오른어깨, 
            7-왼팔꿈치, 8-오른팔꿈치, 9-왼손목, 10-오른손목, 11-왼엉덩이, 12-오른엉덩이,
            13-왼무릎, 14-오른무릎, 15-왼발목, 16-오른발목
            
    COCO18: 위 17개 + 1-목 (어깨 중점으로 계산)
    
    Args:
        data: COCO17 형태의 키포인트 데이터
        
    Returns:
        COCO18 형태로 변환된 키포인트 데이터
    """
    result = {}
    
    for person_id, frames in data.items():
        result[person_id] = {}
        
        for frame_num, frame_data in frames.items():
            keypoints = frame_data["keypoints"]
            
            # 새로운 키포인트 리스트 생성 (18개)
            new_keypoints = []
            
            # 0: 코 (그대로 유지)
            new_keypoints.append(keypoints[0])
            
            # 1: 목 (어깨 중점으로 계산) - 새로 추가
            left_shoulder = keypoints[5]   # 왼어깨
            right_shoulder = keypoints[6]  # 오른어깨
            
            if left_shoulder[2] > 0 and right_shoulder[2] > 0:  # 둘 다 유효한 경우
                neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
                neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
                neck_conf = min(left_shoulder[2], right_shoulder[2])  # 더 낮은 신뢰도 사용
                new_keypoints.append([neck_x, neck_y, neck_conf])
            elif left_shoulder[2] > 0:  # 왼어깨만 유효한 경우
                new_keypoints.append([left_shoulder[0], left_shoulder[1] - 20, left_shoulder[2] * 0.7])
            elif right_shoulder[2] > 0:  # 오른어깨만 유효한 경우  
                new_keypoints.append([right_shoulder[0], right_shoulder[1] - 20, right_shoulder[2] * 0.7])
            else:  # 둘 다 무효한 경우
                new_keypoints.append([0, 0, 0])
            
            # 2-17: 나머지 키포인트들 (COCO17의 1-16에 해당)
            for i in range(1, 17):
                new_keypoints.append(keypoints[i])
            
            result[person_id][frame_num] = {"keypoints": new_keypoints}
    
    return result


def normalize_keypoints(data, method="frame", frame_width=1920, frame_height=1080):
    """
    키포인트 데이터를 정규화합니다.
    
    Args:
        data: 키포인트 데이터
        method: 정규화 방법 ("bbox": 바운딩박스 기준, "frame": 프레임 크기 기준)
        frame_width: 프레임 너비 (기본값: 1920)
        frame_height: 프레임 높이 (기본값: 1080)
        
    Returns:
        정규화된 키포인트 데이터
    """
    result = {}
    
    for person_id, frames in data.items():
        result[person_id] = {}
        
        if method == "bbox":
            # 해당 사람의 모든 프레임에서 바운딩박스 계산
            all_x = []
            all_y = []
            
            for frame_data in frames.values():
                for keypoint in frame_data["keypoints"]:
                    if keypoint[2] > 0:  # 유효한 키포인트만
                        all_x.append(keypoint[0])
                        all_y.append(keypoint[1])
            
            if all_x and all_y:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                
                # 바운딩박스 크기
                bbox_width = max_x - min_x if max_x > min_x else 1
                bbox_height = max_y - min_y if max_y > min_y else 1
                
                # 각 프레임 정규화
                for frame_num, frame_data in frames.items():
                    normalized_keypoints = []
                    
                    for keypoint in frame_data["keypoints"]:
                        if keypoint[2] > 0:  # 유효한 키포인트
                            norm_x = (keypoint[0] - min_x) / bbox_width
                            norm_y = (keypoint[1] - min_y) / bbox_height
                            normalized_keypoints.append([norm_x, norm_y, keypoint[2]])
                        else:
                            normalized_keypoints.append([0, 0, 0])
                    
                    result[person_id][frame_num] = {"keypoints": normalized_keypoints}
            else:
                # 유효한 키포인트가 없으면 원본 유지
                result[person_id] = frames
                
        elif method == "frame":
            # 프레임 크기 기준 정규화 (1920x1080)
            for frame_num, frame_data in frames.items():
                normalized_keypoints = []
                
                for keypoint in frame_data["keypoints"]:
                    if keypoint[2] > 0:  # 유효한 키포인트
                        norm_x = keypoint[0] / frame_width
                        norm_y = keypoint[1] / frame_height
                        normalized_keypoints.append([norm_x, norm_y, keypoint[2]])
                    else:
                        normalized_keypoints.append([0, 0, 0])
                
                result[person_id][frame_num] = {"keypoints": normalized_keypoints}
    
    return result


def standardize_keypoints(sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    키포인트 시퀀스를 표준화합니다 (평균=0, 표준편차=1).
    
    Args:
        sequences: 슬라이딩 윈도우로 분할된 시퀀스 리스트
        
    Returns:
        표준화된 시퀀스 리스트
    """
    if not sequences:
        return sequences
    
    # 모든 시퀀스의 키포인트 좌표 수집 (x, y만, confidence는 제외)
    all_coordinates = []
    
    for seq_data in sequences:
        sequence = seq_data["sequence"]  # (window_size, num_keypoints, 3)
        
        # 유효한 키포인트의 x, y 좌표만 수집
        for frame in sequence:
            for keypoint in frame:
                if keypoint[2] > 0:  # confidence > 0인 경우만
                    all_coordinates.extend([keypoint[0], keypoint[1]])
    
    if not all_coordinates:
        return sequences
    
    # 평균과 표준편차 계산
    coordinates_array = np.array(all_coordinates)
    mean_coord = np.mean(coordinates_array)
    std_coord = np.std(coordinates_array)
    
    # 표준편차가 0인 경우 방지
    if std_coord == 0:
        std_coord = 1
    
    # 각 시퀀스 표준화
    standardized_sequences = []
    
    for seq_data in sequences:
        sequence = seq_data["sequence"].copy()  # (window_size, num_keypoints, 3)
        
        # x, y 좌표만 표준화 (confidence는 그대로 유지)
        for frame_idx in range(sequence.shape[0]):
            for kp_idx in range(sequence.shape[1]):
                if sequence[frame_idx, kp_idx, 2] > 0:  # 유효한 키포인트만
                    # x 좌표 표준화
                    sequence[frame_idx, kp_idx, 0] = (sequence[frame_idx, kp_idx, 0] - mean_coord) / std_coord
                    # y 좌표 표준화  
                    sequence[frame_idx, kp_idx, 1] = (sequence[frame_idx, kp_idx, 1] - mean_coord) / std_coord
        
        # 표준화된 시퀀스 추가
        new_seq_data = seq_data.copy()
        new_seq_data["sequence"] = sequence
        standardized_sequences.append(new_seq_data)
    
    return standardized_sequences


def create_test_data():
    """테스트용 데이터 생성 (COCO17 형식)"""
    test_data = {
        "person_1": {}
    }
    
    # 50프레임의 테스트 데이터 생성 (17개 키포인트)
    for frame in range(50):
        # 간단한 움직임 패턴 (원형 움직임)
        angle = frame * 0.1
        
        # 17개 키포인트 생성 (COCO17 형식)
        keypoints = []
        
        # 0: 코
        keypoints.append([960 + 10 * np.cos(angle), 400 + 10 * np.sin(angle), 0.9])
        # 1: 왼눈
        keypoints.append([950 + 8 * np.cos(angle), 390 + 8 * np.sin(angle), 0.8])
        # 2: 오른눈  
        keypoints.append([970 + 8 * np.cos(angle), 390 + 8 * np.sin(angle), 0.8])
        # 3: 왼귀
        keypoints.append([940 + 12 * np.cos(angle), 400 + 12 * np.sin(angle), 0.7])
        # 4: 오른귀
        keypoints.append([980 + 12 * np.cos(angle), 400 + 12 * np.sin(angle), 0.7])
        # 5: 왼어깨
        keypoints.append([900 + 30 * np.cos(angle), 500 + 30 * np.sin(angle), 0.8])
        # 6: 오른어깨
        keypoints.append([1020 + 30 * np.cos(angle), 500 + 30 * np.sin(angle), 0.8])
        # 7: 왼팔꿈치
        keypoints.append([850 + 40 * np.cos(angle + 0.5), 600 + 40 * np.sin(angle + 0.5), 0.7])
        # 8: 오른팔꿈치
        keypoints.append([1070 + 40 * np.cos(angle - 0.5), 600 + 40 * np.sin(angle - 0.5), 0.7])
        # 9: 왼손목
        keypoints.append([800 + 50 * np.cos(angle + 1), 700 + 50 * np.sin(angle + 1), 0.6])
        # 10: 오른손목
        keypoints.append([1120 + 50 * np.cos(angle - 1), 700 + 50 * np.sin(angle - 1), 0.6])
        # 11: 왼엉덩이
        keypoints.append([920 + 20 * np.cos(angle), 800 + 20 * np.sin(angle), 0.8])
        # 12: 오른엉덩이
        keypoints.append([1000 + 20 * np.cos(angle), 800 + 20 * np.sin(angle), 0.8])
        # 13: 왼무릎
        keypoints.append([910 + 25 * np.cos(angle + 0.3), 950 + 25 * np.sin(angle + 0.3), 0.7])
        # 14: 오른무릎
        keypoints.append([1010 + 25 * np.cos(angle - 0.3), 950 + 25 * np.sin(angle - 0.3), 0.7])
        # 15: 왼발목
        keypoints.append([905 + 30 * np.cos(angle + 0.6), 1050 + 30 * np.sin(angle + 0.6), 0.6])
        # 16: 오른발목
        keypoints.append([1015 + 30 * np.cos(angle - 0.6), 1050 + 30 * np.sin(angle - 0.6), 0.6])
        
        test_data["person_1"][str(frame)] = {
            "keypoints": keypoints
        }
    
    return test_data


def test_sliding_window():
    """슬라이딩 윈도우 테스트"""
    print("=== 슬라이딩 윈도우 시퀀스 분할 테스트 ===")
    
    # 테스트 데이터 생성
    test_data = create_test_data()
    print(f"원본 데이터: {len(test_data['person_1'])} 프레임")
    
    # 슬라이딩 윈도우 적용
    sequences = sliding_window_sequence_split(test_data, window_size=10, stride=5)
    
    print(f"생성된 시퀀스 개수: {len(sequences)}")
    for i, seq_data in enumerate(sequences):
        print(f"시퀀스 {i}: 프레임 {seq_data['start_frame']}-{seq_data['end_frame']}, 크기: {seq_data['sequence'].shape}")


def test_full_transform_pipeline():
    """전체 변환 파이프라인 테스트"""
    print("=== 전체 변환 파이프라인 테스트 ===")
    
    # 1. 테스트 데이터 생성
    test_data = create_test_data()
    print(f"1. 원본 데이터: {len(test_data['person_1'])} 프레임")
    
    # 2. COCO18 변환
    coco18_data = convert_to_coco18(test_data)
    original_keypoints = len(test_data['person_1']['0']['keypoints'])
    new_keypoints = len(coco18_data['person_1']['0']['keypoints'])
    print(f"2. COCO18 변환: {original_keypoints} → {new_keypoints} 키포인트")
    
    # 3. 정규화 (1920x1080 해상도 기준)
    normalized_data = normalize_keypoints(coco18_data, method="frame", frame_width=1920, frame_height=1080)
    print("3. 프레임 해상도 기준 정규화 완료 (1920x1080)")
    
    # 4. 슬라이딩 윈도우 분할
    sequences = sliding_window_sequence_split(normalized_data, window_size=10, stride=5)
    print(f"4. 슬라이딩 윈도우 분할: {len(sequences)}개 시퀀스 생성")
    
    # 5. 표준화
    standardized_sequences = standardize_keypoints(sequences)
    print("5. 표준화 완료")
    
    # 결과 출력
    print(f"\n최종 결과:")
    print(f"- 총 시퀀스 개수: {len(standardized_sequences)}")
    if standardized_sequences:
        first_seq = standardized_sequences[0]["sequence"]
        print(f"- 각 시퀀스 크기: {first_seq.shape}")
        print(f"- 첫 번째 시퀀스 첫 프레임 첫 키포인트: {first_seq[0, 0]}")


if __name__ == "__main__":
    # 개별 테스트
    # test_sliding_window()
    
    # 전체 파이프라인 테스트
    test_full_transform_pipeline()