import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import re

def count_video_frames(video_path):
    """비디오 파일의 프레임 수를 계산합니다."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return 0

def parse_xml_for_theft_frames(xml_path):
    """XML 파일에서 도난(theft) 시작과 끝 프레임을 추출합니다."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        theft_start = None
        theft_end = None
        
        # theft_start와 theft_end 트랙을 찾습니다
        for track in root.findall('.//track'):
            label = track.get('label')
            if label == 'theft_start':
                # 첫 번째 박스의 프레임 번호를 가져옵니다
                box = track.find('box')
                if box is not None:
                    frame_attr = box.get('frame')
                    if frame_attr is not None:
                        theft_start = int(frame_attr)
            elif label == 'theft_end':
                # 첫 번째 박스의 프레임 번호를 가져옵니다
                box = track.find('box')
                if box is not None:
                    frame_attr = box.get('frame')
                    if frame_attr is not None:
                        theft_end = int(frame_attr)
        
        return theft_start, theft_end
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return None, None

def find_corresponding_video(xml_path, video_dir):
    """XML 파일명에 대응하는 비디오 파일을 찾습니다."""
    xml_name = Path(xml_path).stem
    
    # 비디오 확장자들
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    for ext in video_extensions:
        video_path = os.path.join(video_dir, xml_name + ext)
        if os.path.exists(video_path):
            return video_path
    
    # 직접 찾기가 안되면 유사한 이름의 파일을 찾습니다
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    # XML 파일명과 가장 유사한 비디오 파일을 찾습니다
    for video_file in video_files:
        video_name = Path(video_file).stem
        if xml_name in video_name or video_name in xml_name:
            return video_file
    
    return None

def count_frames_in_directory(directory):
    """디렉토리의 모든 비디오 파일 프레임 수를 계산합니다."""
    total_frames = 0
    video_count = 0
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, file)
                frames = count_video_frames(video_path)
                total_frames += frames
                video_count += 1
                print(f"  {file}: {frames} frames")
    
    return total_frames, video_count

def analyze_shoplifting_data():
    """도난 탐지 데이터의 정상/이상 프레임을 분석합니다."""
    
    # 경로 설정
    normal_video_dir = "D:\\AI-HUB_shoping"
    abnormal_video_dir = "D:\\AI-HUB_shoplifting"
    
    print("="*60)
    print("도난 탐지 데이터 프레임 분석")
    print("="*60)
    
    # 1. 정상 영상 프레임 개수 계산
    print("\n1. 정상 영상 프레임 개수 계산 중...")
    print("-"*40)
    
    if os.path.exists(normal_video_dir):
        normal_frames, normal_video_count = count_frames_in_directory(normal_video_dir)
        print(f"\n정상 영상 총 개수: {normal_video_count}개")
        print(f"정상 영상 총 프레임 수: {normal_frames:,}개")
    else:
        normal_frames = 0
        normal_video_count = 0
        print(f"정상 영상 디렉토리를 찾을 수 없습니다: {normal_video_dir}")
    
    # 2. 이상 영상에서 정상/이상 프레임 분석
    print("\n\n2. 이상 영상에서 정상/이상 프레임 분석 중...")
    print("-"*40)
    
    if not os.path.exists(abnormal_video_dir):
        print(f"이상 영상 디렉토리를 찾을 수 없습니다: {abnormal_video_dir}")
        return
    
    # XML 파일들을 찾습니다
    xml_files = []
    for root, dirs, files in os.walk(abnormal_video_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    print(f"발견된 XML 파일 개수: {len(xml_files)}개")
    
    total_abnormal_frames = 0  # 도난 프레임
    total_normal_in_abnormal = 0  # 이상 영상 내 정상 프레임
    processed_videos = 0
    
    for xml_path in xml_files:
        print(f"\n분석 중: {Path(xml_path).name}")
        
        # XML에서 도난 프레임 정보 추출
        theft_start, theft_end = parse_xml_for_theft_frames(xml_path)
        
        if theft_start is None or theft_end is None:
            print(f"  ⚠️ 도난 프레임 정보를 찾을 수 없습니다.")
            continue
        
        # 대응하는 비디오 파일 찾기
        video_path = find_corresponding_video(xml_path, abnormal_video_dir)
        
        if video_path is None:
            print(f"  ⚠️ 대응하는 비디오 파일을 찾을 수 없습니다.")
            continue
        
        # 비디오 총 프레임 수 계산
        total_video_frames = count_video_frames(video_path)
        
        if total_video_frames == 0:
            print(f"  ⚠️ 비디오를 읽을 수 없습니다.")
            continue
        
        # 도난 프레임과 정상 프레임 계산
        abnormal_frames = theft_end - theft_start + 1
        normal_frames_in_video = total_video_frames - abnormal_frames
        
        total_abnormal_frames += abnormal_frames
        total_normal_in_abnormal += normal_frames_in_video
        processed_videos += 1
        
        print(f"  📹 비디오: {Path(video_path).name}")
        print(f"  📊 총 프레임: {total_video_frames}")
        print(f"  🚨 도난 프레임: {theft_start}~{theft_end} ({abnormal_frames}개)")
        print(f"  ✅ 정상 프레임: {normal_frames_in_video}개")
    
    # 3. 최종 결과 출력
    print("\n\n" + "="*60)
    print("최종 분석 결과")
    print("="*60)
    
    total_normal_frames = normal_frames + total_normal_in_abnormal
    
    print(f"\n📊 처리된 비디오 통계:")
    print(f"  - 정상 영상: {normal_video_count}개")
    print(f"  - 이상 영상: {processed_videos}개")
    print(f"  - 총 영상: {normal_video_count + processed_videos}개")
    
    print(f"\n📈 프레임 통계:")
    print(f"  - 순수 정상 영상 프레임: {normal_frames:,}개")
    print(f"  - 이상 영상 내 정상 프레임: {total_normal_in_abnormal:,}개")
    print(f"  - 총 정상 프레임: {total_normal_frames:,}개")
    print(f"  - 총 이상(도난) 프레임: {total_abnormal_frames:,}개")
    print(f"  - 전체 프레임: {total_normal_frames + total_abnormal_frames:,}개")
    
    print(f"\n📊 비율:")
    if total_normal_frames + total_abnormal_frames > 0:
        normal_ratio = (total_normal_frames / (total_normal_frames + total_abnormal_frames)) * 100
        abnormal_ratio = (total_abnormal_frames / (total_normal_frames + total_abnormal_frames)) * 100
        print(f"  - 정상 프레임 비율: {normal_ratio:.2f}%")
        print(f"  - 이상 프레임 비율: {abnormal_ratio:.2f}%")
        print(f"  - 정상:이상 비율 = {total_normal_frames}:{total_abnormal_frames}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    analyze_shoplifting_data()
