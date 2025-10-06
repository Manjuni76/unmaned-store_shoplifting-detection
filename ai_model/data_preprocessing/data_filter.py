# 데이터 신뢰도 필터링
# 데이터 후처리(선형 보간법, n-frame 스무딩)

import numpy as np
from typing import Dict, Any, List

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
# 
# 각 키포인트는 [x좌표, y좌표, 신뢰도] 형태의 리스트

def filter_low_confidence(data, conf_threshold=0.3):
    # filter 결과 저장용 딕셔너리
    filtered_data = {}

    # 사람 id 별로 순회 .items()로 data 딕셔너리 key, value 불러옴
    for person_id, frames in data.items():
        # filtered_data 딕셔너리 생성
        filtered_data[person_id] = {}
        
        # 프레임 별로 딕셔너리 순회하면서 딕셔너리 값 불러오기
        for frame_num, frame_data in frames.items():
            # keypoints 리스트 가져오기 (각 키포인트는 [x, y, confidence] 형태)
            keypoints = frame_data.get("keypoints", [])
            filtered_keypoints = []
            
            # 각 키포인트 검사 (리스트 형태로 처리)
            for keypoint in keypoints:
                x, y, confidence = keypoint[0], keypoint[1], keypoint[2]      
                # 필터링 조건 확인: x > 0, y > 0, confidence > 임계값
                if x > 0 and y > 0 and confidence > conf_threshold:
                    # 조건을 만족하면 그대로 유지
                    filtered_keypoints.append([x, y, confidence])
                else:
                    # 조건을 만족하지 않으면 [0, 0, 0]으로 설정 (무효 처리)
                    filtered_keypoints.append([0, 0, 0])
            
            # 필터링된 데이터 저장
            filtered_data[person_id][frame_num] = {"keypoints": filtered_keypoints}
    
    return filtered_data


def linear_interpolation(data, max_gap=8):
    """
    필터링된 키포인트들을 선형 보간법으로 복원합니다.
    
    Args:
        data: 필터링된 키포인트 데이터
        max_gap: 보간할 최대 프레임 간격 (기본값: 8)
        
    Returns:
        보간된 키포인트 데이터
    """
    # 선형 보간법 결과 저장용 딕셔너리 생성
    result = {}
    
    for person_id, frames in data.items():
        result[person_id] = {}
        
        # 프레임을 번호 순서로 정렬 (문자열을 정수로 변환하여 정렬)
        sorted_frames = sorted(frames.keys(), key=int)
        
        # 프레임 순서대로 처리
        for frame_num in sorted_frames:
            frame_data = frames[frame_num] #프레임 순서대로 처리하지만 실제 값음 frames 안에 있으므로 frames[frame_num]임
            keypoints = frame_data.get("keypoints", [])
            interpolated_keypoints = [] # 보간된 키포인트 저장소
            
            # 각 키포인트 위치별로 보간 처리
            for keypoint_idx in range(len(keypoints)):
                current_keypoint = keypoints[keypoint_idx]
                
                # 현재 키포인트가 유효한 경우 (confidence > 0)
                if current_keypoint[0] > 0 and current_keypoint[1] > 0 and current_keypoint[2] > 0:
                    interpolated_keypoints.append(current_keypoint)
                else:
                    # 무효한 키포인트인 경우 선형 보간 시도
                    prev_valid_frame = None  # 이전 유효 프레임
                    next_valid_frame = None  # 다음 유효 프레임
                    
                    # 이전 프레임들 중에서 유효한 키포인트 찾기
                    for prev_frame in reversed(sorted_frames):
                        if int(prev_frame) < int(frame_num):
                            prev_keypoint = frames[prev_frame]["keypoints"][keypoint_idx]
                            if prev_keypoint[0] > 0 and prev_keypoint[1] > 0 and prev_keypoint[2] > 0:  # 유효한 키포인트
                                prev_valid_frame = (int(prev_frame), prev_keypoint)
                                break
                    
                    # 다음 프레임들 중에서 유효한 키포인트 찾기
                    for next_frame in sorted_frames:
                        if int(next_frame) > int(frame_num):
                            next_keypoint = frames[next_frame]["keypoints"][keypoint_idx]
                            if next_keypoint[0] > 0 and next_keypoint[1] > 0 and next_keypoint[2] > 0:
                                next_valid_frame = (int(next_frame), next_keypoint)
                                break

                    # 선형 보간 적용
                    if prev_valid_frame and next_valid_frame:
                        frame_gap = next_valid_frame[0] - prev_valid_frame[0]
                        if frame_gap <= max_gap:  # 간격이 너무 크지 않은 경우만 보간
                            # 보간 비율 계산
                            ratio = (int(frame_num) - prev_valid_frame[0]) / frame_gap
                            
                            # x, y 좌표 보간
                            x = prev_valid_frame[1][0] + (next_valid_frame[1][0] - prev_valid_frame[1][0]) * ratio
                            y = prev_valid_frame[1][1] + (next_valid_frame[1][1] - prev_valid_frame[1][1]) * ratio
                            confidence = 0.5  # 보간된 키포인트의 신뢰도
                            interpolated_keypoint = [x, y, confidence]
                        else:
                            interpolated_keypoint = [0, 0, 0]  # 간격이 너무 크면 보간하지 않음
                    else:
                        interpolated_keypoint = [0, 0, 0]  # 이전/다음 유효 프레임이 없으면 보간하지 않음
                    
                    interpolated_keypoints.append(interpolated_keypoint)
            
            # 보간된 데이터 저장
            result[person_id][frame_num] = {"keypoints": interpolated_keypoints}
    
    return result


def n_frame_smoothing(data: Dict[str, Dict[str, Dict]], window_size: int = 8) -> Dict[str, Dict[str, Dict]]:
    """
    n-frame 스무딩을 적용하여 키포인트 움직임을 부드럽게 만듭니다.
    
    Args:
        data: 보간된 키포인트 데이터
        window_size: 스무딩 윈도우 크기 (기본값: 8, 홀수 권장)
        
    Returns:
        스무딩된 키포인트 데이터
    """
    result = {}
    
    for person_id, frames in data.items():
        result[person_id] = {}
        
        # 프레임을 번호 순서로 정렬
        sorted_frames = sorted(frames.keys(), key=int)
        
        for frame_num in sorted_frames:
            frame_data = frames[frame_num]
            keypoints = frame_data.get("keypoints", [])
            #스무딩 처리된 키포인트 저장소 생성
            smoothed_keypoints = []
            
            # 각 키포인트별로 스무딩 처리
            for keypoint_idx in range(len(keypoints)):
                current_keypoint = keypoints[keypoint_idx]
                
                # 현재 키포인트가 무효한 경우 그대로 유지
                if current_keypoint[2] == 0:  # confidence가 0인 경우
                    smoothed_keypoints.append(current_keypoint)
                else:
                    # 윈도우 사이즈 반 크기 계산
                    half_window = window_size // 2
                    # 현재 프레임 인덱스
                    current_frame_idx = int(frame_num)
                    
                    window_keypoints = []
                    for i in range( -half_window, half_window+1):
                        target_frame = str(current_frame_idx + i)
                        if target_frame in frames:
                            target_keypoint = frames[target_frame]["keypoints"][keypoint_idx]
                            if target_keypoint[0] > 0 and target_keypoint[1] > 0 and target_keypoint[2] > 0:
                                window_keypoints.append(target_keypoint)
                    if len(window_keypoints) > 0:
                        avg_x = sum(kp[0] for kp in window_keypoints) / len(window_keypoints)
                        avg_y = sum(kp[1] for kp in window_keypoints) / len(window_keypoints)
                        avg_conf = sum(kp[2] for kp in window_keypoints) / len(window_keypoints)
                        smoothed_keypoint = [avg_x, avg_y, avg_conf]
                    else:
                        smoothed_keypoint = current_keypoint  # 스무딩할 데이터가 없으면 원본 유지
                    
                    smoothed_keypoints.append(smoothed_keypoint)
            
            # 스무딩된 데이터 저장
            result[person_id][frame_num] = {"keypoints": smoothed_keypoints}
    
    return result
def create_interpolation_test_data():
    """선형 보간법 테스트용 데이터 생성"""
    test_data = {
        "person_1": {
            "0": {"keypoints": [[100, 150, 0.8], [90, 140, 0.7]]},   # 유효
            "1": {"keypoints": [[0, 0, 0], [0, 0, 0]]},              # 무효 (보간 대상)
            "2": {"keypoints": [[0, 0, 0], [0, 0, 0]]},              # 무효 (보간 대상)  
            "3": {"keypoints": [[0, 0, 0], [100, 160, 0.6]]},       # 부분 무효
            "4": {"keypoints": [[130, 180, 0.9], [110, 170, 0.8]]}, # 유효
        }
    }
    return test_data


def create_smoothing_test_data():
    """스무딩 테스트용 데이터 생성 (노이즈가 있는 데이터)"""
    test_data = {
        "person_1": {
            "0": {"keypoints": [[100, 150, 0.8], [90, 140, 0.7]]},
            "1": {"keypoints": [[105, 155, 0.8], [95, 145, 0.7]]},  # 작은 변화
            "2": {"keypoints": [[120, 180, 0.8], [110, 170, 0.7]]}, # 급격한 변화 (노이즈)
            "3": {"keypoints": [[108, 158, 0.8], [98, 148, 0.7]]},  # 다시 정상 범위
            "4": {"keypoints": [[110, 160, 0.8], [100, 150, 0.7]]}, # 정상
            "5": {"keypoints": [[85, 135, 0.8], [80, 130, 0.7]]},   # 급격한 변화 (노이즈)
            "6": {"keypoints": [[112, 162, 0.8], [102, 152, 0.7]]}, # 다시 정상 범위
        }
    }
    return test_data


def test_smoothing():
    """스무딩 함수 테스트"""
    print("=== n-frame 스무딩 테스트 ===")
    
    # 테스트 데이터 생성
    test_data = create_smoothing_test_data()
    print("\n스무딩 전 데이터:")
    for person_id, frames in test_data.items():
        print(f"  {person_id}:")
        for frame_num, frame_data in frames.items():
            keypoints = frame_data["keypoints"]
            print(f"    프레임 {frame_num}: {keypoints}")
    
    # 스무딩 적용
    smoothed_data = n_frame_smoothing(test_data, window_size=5)
    
    print("\n스무딩 후 데이터:")
    for person_id, frames in smoothed_data.items():
        print(f"  {person_id}:")
        for frame_num, frame_data in frames.items():
            keypoints = frame_data["keypoints"]
            print(f"    프레임 {frame_num}: {keypoints}")


def test_full_pipeline():
    """전체 파이프라인 테스트 (필터링 → 보간 → 스무딩)"""
    print("=== 전체 파이프라인 테스트 ===")
    
    # 1. 원본 데이터 (노이즈 포함)
    original_data = {
        "person_1": {
            "0": {"keypoints": [[100, 150, 0.8], [90, 140, 0.7]]},
            "1": {"keypoints": [[50, 80, 0.1], [0, 0, 0]]},        # 낮은 신뢰도 + 무효
            "2": {"keypoints": [[0, 0, 0], [0, 0, 0]]},            # 무효
            "3": {"keypoints": [[130, 200, 0.9], [120, 190, 0.8]]}, # 급격한 변화
            "4": {"keypoints": [[110, 160, 0.8], [100, 150, 0.7]]}, # 정상
        }
    }
    
    print("\n1. 원본 데이터:")
    for person_id, frames in original_data.items():
        for frame_num, frame_data in frames.items():
            keypoints = frame_data["keypoints"]
            print(f"  프레임 {frame_num}: {keypoints}")
    
    # 2. 필터링 적용
    filtered_data = filter_low_confidence(original_data, conf_threshold=0.3)
    print("\n2. 필터링 후:")
    for person_id, frames in filtered_data.items():
        for frame_num, frame_data in frames.items():
            keypoints = frame_data["keypoints"]
            print(f"  프레임 {frame_num}: {keypoints}")
    
    # 3. 선형 보간 적용
    interpolated_data = linear_interpolation(filtered_data, max_gap=8)
    print("\n3. 보간 후:")
    for person_id, frames in interpolated_data.items():
        for frame_num, frame_data in frames.items():
            keypoints = frame_data["keypoints"]
            print(f"  프레임 {frame_num}: {keypoints}")
    
    # 4. 스무딩 적용
    smoothed_data = n_frame_smoothing(interpolated_data, window_size=3)
    print("\n4. 스무딩 후:")
    for person_id, frames in smoothed_data.items():
        for frame_num, frame_data in frames.items():
            keypoints = frame_data["keypoints"]
            print(f"  프레임 {frame_num}: {keypoints}")

def test_interpolation():
    """선형 보간법 테스트"""
    print("=== 선형 보간법 테스트 ===")
    
    # 테스트 데이터 생성
    test_data = create_interpolation_test_data()
    print("\n보간 전 데이터:")
    for person_id, frames in test_data.items():
        print(f"  {person_id}:")
        for frame_num, frame_data in frames.items():
            keypoints = frame_data["keypoints"]
            print(f"    프레임 {frame_num}: {keypoints}")
    
    # 선형 보간 적용
    interpolated_data = linear_interpolation(test_data, max_gap=8)
    
    print("\n보간 후 데이터:")
    for person_id, frames in interpolated_data.items():
        print(f"  {person_id}:")
        for frame_num, frame_data in frames.items():
            keypoints = frame_data["keypoints"]
            print(f"    프레임 {frame_num}: {keypoints}")


if __name__ == "__main__":
    # 각각 테스트
    # test_filter()          # 필터링만 테스트
    # test_interpolation()   # 보간만 테스트
    # test_smoothing()       # 스무딩만 테스트
    
    # 전체 파이프라인 테스트
    test_full_pipeline()
        