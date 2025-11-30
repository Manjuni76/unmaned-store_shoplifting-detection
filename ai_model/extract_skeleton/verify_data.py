import cv2
import json
import numpy as np
from pathlib import Path
import argparse
import time
from tqdm import tqdm

# (!!!) 18개 관절(COCO+Neck) 기준 연결선 (!!!)
# (ai_model/dataset.py의 COCO-18 변환 기준)
# 1번(Neck)이 중심이 됩니다.
COCO_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (0, 1), (0, 14), (14, 16), (0, 15), (15, 17)
]
# 0: Nose, 1: Neck (중심)
# 2: RShoulder, 3: RElbow, 4: RWrist
# 5: LShoulder, 6: LElbow, 7: LWrist
# 8: RHip, 9: RKnee, 10: RAnkle
# 11: LHip, 12: LKnee, 13: LAnkle
# 14: REye, 15: LEye, 16: REar, 17: LEar

def draw_skeleton(frame, keypoints, pairs, threshold=0.1):
    """검은색 프레임에 스켈레톤을 그립니다."""
    points = {}
    valid_kp_count = 0
    
    for i, kp in enumerate(keypoints):
        x, y, conf = kp
        if conf > threshold:
            # (!!!) 해상도에 맞게 좌표 스케일링 (!!!)
            # (1920x1080 원본 좌표 기준이므로 그대로 사용)
            px, py = int(x), int(y)
            
            # (!!!) 프레임 바깥으로 나간 좌표는 그리지 않음 (!!!)
            if 0 < px < frame.shape[1] and 0 < py < frame.shape[0]:
                points[i] = (px, py)
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1) # 녹색 점
                valid_kp_count += 1
    
    for pair in pairs:
        if pair[0] in points and pair[1] in points:
            cv2.line(frame, points[pair[0]], points[pair[1]], (0, 0, 255), 2) # 빨간색 선
            
    return frame, valid_kp_count

def visualize_json_only(skeleton_json_path, fps=3.0, width=1920, height=1080):
    """JSON 파일만 읽어서 스켈레톤 비디오를 재생합니다."""
        
    if not skeleton_json_path.exists():
        print(f"[오류] 스켈레톤 JSON 파일을 찾을 수 없습니다: {skeleton_json_path}")
        return

    # 1. 스켈레톤 데이터 로드
    with open(skeleton_json_path, 'r') as f:
        skeleton_data = json.load(f)
    
    person_1_skeletons = skeleton_data.get("person_1", {})
    if not person_1_skeletons:
        print("[오류] JSON 파일에 'person_1' 데이터가 없습니다.")
        return

    # (!!!) 프레임 정렬 (JSON은 순서 보장 안 됨) (!!!)
    # 프레임 번호(str)를 정수(int)로 변환하여 정렬
    sorted_frames = sorted(person_1_skeletons.keys(), key=int)
    total_frames = len(sorted_frames)
    
    # 3.0 FPS 기준 딜레이
    delay = int(1000 / fps) 

    print("--- JSON 시각화 시작 ---")
    print(f"스켈레톤: {skeleton_json_path.name}")
    print(f"총 프레임: {total_frames} (FPS: {fps})")
    print("ESC 키를 누르면 종료됩니다.")
    
    frame_idx = 0
    for frame_key in tqdm(sorted_frames, desc="Visualizing"):
        
        # 1. 검은색 도화지(Canvas) 생성
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        keypoints = person_1_skeletons[frame_key].get("keypoints")
        
        valid_kp_count = 0
        if keypoints:
            # 2. 도화지에 스켈레톤 그리기
            canvas, valid_kp_count = draw_skeleton(canvas, keypoints, COCO_PAIRS, threshold=0.1)
        
        if valid_kp_count == 0:
            # 빈 프레임 (0으로 패딩됨)
            cv2.putText(canvas, "EMPTY FRAME (PADDING)", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv2.putText(canvas, f"Frame: {frame_key}", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # (!!!) 프레임 크기를 화면에 맞게 조절 (예: 1/2 크기) (!!!)
        canvas_resized = cv2.resize(canvas, (width // 2, height // 2))
        
        cv2.imshow('JSON Skeleton Visualizer', canvas_resized)
        
        # ESC 키로 종료
        if cv2.waitKey(delay) & 0xFF == 27:
            break
            
        frame_idx += 1

    cv2.destroyAllWindows()
    print("--- 시각화 종료 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skeleton JSON Data Visualizer (Video-less)')
    parser.add_argument('--json', type=str, required=True, help='V6로 추출한 스켈레톤 JSON 파일 경로')
    parser.add_argument('--fps', type=float, default=3.0, help='재생 속도 (FPS) (3.0 권장)')
    parser.add_argument('--width', type=int, default=1920, help='원본 비디오 너비')
    parser.add_argument('--height', type=int, default=1080, help='원본 비디오 높이')
    args = parser.parse_args()

    skeleton_json_path = Path(args.json)
    
    visualize_json_only(skeleton_json_path, args.fps, args.width, args.height)