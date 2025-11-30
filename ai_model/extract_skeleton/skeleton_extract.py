"""
정상 데이터: XML bbox 기반 스켈레톤 추출 (ID가 있는 모든 인물)
이상 데이터: 단순 1인 스켈레톤 추출
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import argparse
import multiprocessing as mp
import xml.etree.ElementTree as ET


class SkeletonExtractorWithXML:
    def __init__(self, model_name='yolo11x-pose.pt', conf_threshold=0.5, kpt_threshold=0.3, device='cuda'):
        """
        XML bbox 기반 또는 단순 추출 모드
        """
        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self.device = device
        
        print(f"[INFO] Loading {model_name}...")
        self.model = YOLO(model_name)
        
        if device == 'cuda' and torch.cuda.is_available():
            self.model.to('cuda')
            print(f"[INFO] Using GPU")
        else:
            self.device = 'cpu'
            print(f"[INFO] Using CPU")
        
        print(f"[SUCCESS] Model loaded")
    
    def parse_xml_bbox(self, xml_path):
        """
        XML 파일에서 ID가 있는 모든 인물의 bbox 시퀀스 추출 (ID 없는 인물 제외)
        
        Returns:
            dict: {person_id: {frame_idx: [xtl, ytl, xbr, ybr], ...}, ...}
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            bbox_map_all_persons = {}
            
            # 모든 track을 순회
            for track in root.findall('.//track'):
                # 각 track의 box들을 확인
                for box in track.findall('box'):
                    # ID 속성 확인
                    id_attr = box.find("attribute[@name='ID']")
                    if id_attr is not None: # ID 속성이 있는 경우에만 처리 (점주 등 ID 없는 인물 제외)
                        try:
                            box_id = int(id_attr.text)
                            
                            # 해당 ID의 딕셔너리가 없으면 생성
                            if box_id not in bbox_map_all_persons:
                                bbox_map_all_persons[box_id] = {}
                                
                            # 프레임 번호와 bbox 좌표 추출
                            frame_idx = int(box.get('frame'))
                            xtl = float(box.get('xtl'))
                            ytl = float(box.get('ytl'))
                            xbr = float(box.get('xbr'))
                            ybr = float(box.get('ybr'))
                            
                            bbox_map_all_persons[box_id][frame_idx] = [xtl, ytl, xbr, ybr]
                        except (ValueError, TypeError):
                            continue
            
            return bbox_map_all_persons
        except Exception as e:
            print(f"[WARNING] XML parsing failed: {e}")
            return {}
    
    def extract_pose_in_bbox(self, frame, bbox):
        """
        특정 bbox 영역의 사람 포즈 추출
        
        Args:
            frame: 비디오 프레임
            bbox: [xtl, ytl, xbr, ybr]
        
        Returns:
            keypoints: [[x, y, conf], ...] or None
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False, device=self.device)
        
        if len(results) == 0 or results[0].keypoints is None or results[0].boxes is None:
            return None
        
        if results[0].keypoints.xy.numel() == 0:
            return None
        
        keypoints_all = results[0].keypoints.xy.cpu().numpy()
        confidences_all = results[0].keypoints.conf.cpu().numpy()
        boxes_all = results[0].boxes.xyxy.cpu().numpy()
        
        # bbox와 IoU가 가장 높은 detection 찾기
        xtl, ytl, xbr, ybr = bbox
        best_iou = 0.0
        best_idx = -1
        
        for i, det_box in enumerate(boxes_all):
            x1, y1, x2, y2 = det_box
            
            # IoU 계산
            xi1 = max(xtl, x1)
            yi1 = max(ytl, y1)
            xi2 = min(xbr, x2)
            yi2 = min(ybr, y2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (xbr - xtl) * (ybr - ytl)
            box2_area = (x2 - x1) * (y2 - y1)
            union_area = box1_area + box2_area - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        # IoU 0.3 이상이면 매칭 성공
        if best_iou < 0.3:
            return None
        
        # 키포인트 변환
        person_keypoints = []
        for j in range(17):
            if j < len(keypoints_all[best_idx]):
                x, y = keypoints_all[best_idx][j]
                conf = confidences_all[best_idx][j] if j < len(confidences_all[best_idx]) else 0.0
                
                if conf < self.kpt_threshold or x <= 0 or y <= 0:
                    person_keypoints.append([0.0, 0.0, 0.0])
                else:
                    person_keypoints.append([float(x), float(y), float(conf)])
            else:
                person_keypoints.append([0.0, 0.0, 0.0])
        
        return person_keypoints
    
    def extract_single_person_pose(self, frame):
        """
        단순 1인 포즈 추출 (이상 데이터용)
        
        Returns:
            keypoints: [[x, y, conf], ...] or None
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False, device=self.device)
        
        if len(results) == 0 or results[0].keypoints is None or results[0].boxes is None:
            return None
        
        if results[0].keypoints.xy.numel() == 0:
            return None
        
        keypoints = results[0].keypoints.xy.cpu().numpy()
        confidences = results[0].keypoints.conf.cpu().numpy()
        
        # 가장 높은 confidence 사람 선택
        if len(results[0].boxes.conf) > 0:
            best_idx = results[0].boxes.conf.cpu().numpy().argmax()
        else:
            best_idx = 0
        
        # 키포인트 변환
        person_keypoints = []
        for j in range(17):
            if j < len(keypoints[best_idx]):
                x, y = keypoints[best_idx][j]
                conf = confidences[best_idx][j] if j < len(confidences[best_idx]) else 0.0
                
                if conf < self.kpt_threshold or x <= 0 or y <= 0:
                    person_keypoints.append([0.0, 0.0, 0.0])
                else:
                    person_keypoints.append([float(x), float(y), float(conf)])
            else:
                person_keypoints.append([0.0, 0.0, 0.0])
        
        return person_keypoints
    
    def extract_with_xml(self, video_path, xml_path, output_json_path):
        """
        XML bbox 기반 스켈레톤 추출 (정상 데이터용) - ID가 있는 모든 인물
        """
        # XML에서 bbox 로드
        bbox_map_all_persons = self.parse_xml_bbox(xml_path)
        
        if not bbox_map_all_persons:
            print(f"[WARNING] No bbox found with ID in XML for {os.path.basename(str(video_path))}")
            return False
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 모든 ID가 있는 사람에 대해 skeleton_data 초기화
        skeleton_data = {}
        person_ids_to_extract = list(bbox_map_all_persons.keys())
        for pid in person_ids_to_extract:
            skeleton_data[f"person_{pid}"] = {}
        
        print(f"[INFO] Extracting with XML bbox (IDs: {person_ids_to_extract})...")
        pbar = tqdm(total=total_frames, desc=f"XML-based {os.path.basename(str(video_path))}", leave=False, ncols=100)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 추출 대상인 모든 사람에 대해 처리
            for pid in person_ids_to_extract:
                person_key = f"person_{pid}"
                bbox_map_for_person = bbox_map_all_persons[pid]

                if frame_idx in bbox_map_for_person:
                    # 이 프레임에 이 사람의 bbox가 있음
                    bbox = bbox_map_for_person[frame_idx]
                    keypoints = self.extract_pose_in_bbox(frame, bbox)
                    
                    if keypoints is not None:
                        skeleton_data[person_key][str(frame_idx)] = {"keypoints": keypoints}
                    else:
                        skeleton_data[person_key][str(frame_idx)] = {"keypoints": [[0.0, 0.0, 0.0]] * 17}
                else:
                    # 이 프레임에 이 사람의 bbox가 없음 (행동 전/후 등)
                    skeleton_data[person_key][str(frame_idx)] = {"keypoints": [[0.0, 0.0, 0.0]] * 17}
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # JSON 저장
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Saved data for person IDs: {person_ids_to_extract}")
        return True
    
    def extract_simple(self, video_path, output_json_path):
        """
        단순 1인 스켈레톤 추출 (이상 데이터용)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skeleton_data = {"person_1": {}}
        
        print(f"[INFO] Extracting single person (abnormal data)...")
        pbar = tqdm(total=total_frames, desc=f"Simple {os.path.basename(str(video_path))}", leave=False, ncols=100)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints = self.extract_single_person_pose(frame)
            
            if keypoints is not None:
                skeleton_data["person_1"][str(frame_idx)] = {"keypoints": keypoints}
            else:
                skeleton_data["person_1"][str(frame_idx)] = {"keypoints": [[0.0, 0.0, 0.0]] * 17}
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # JSON 저장
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
        
        return True


def load_json_split(json_path):
    """Legacy data_split JSON 로드 (제거됨): 현재는 skeleton_extracted 기반으로 직접 스캔을 권장"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    videos = []
    for item in data.get('normal', []):
        item['category'] = 'normal'
        videos.append(item)
    
    for item in data.get('abnormal', []):
        item['category'] = 'abnormal'
        videos.append(item)
    
    return videos


def process_single_video(args_tuple):
    """
    멀티프로세싱용 비디오 처리
    """
    video_info, output_skeleton_dir, model_name, conf, kpt_conf, device, xml_path_or_none = args_tuple
    
    filename = video_info['filename']
    category = video_info['category']
    
    # 비디오 경로 (full_path 사용)
    video_path = Path(video_info['full_path'])
    
    # 출력 JSON 경로
    skeleton_filename = filename.replace('.mp4', '.json')
    output_json_path = output_skeleton_dir / skeleton_filename
    
    # 이미 존재하면 스킵
    if output_json_path.exists():
        return ('skip', filename, category)
    
    # 비디오 존재 확인
    if not video_path.exists():
        return ('fail_video', filename, category)
    
    # 정상 데이터인데 XML이 없으면 단순 추출로 처리 (에러 아님)
    # (C_2 파일이나 제외된 디렉토리의 경우 xml_path_or_none이 None일 수 있음)
    
    try:
        extractor = SkeletonExtractorWithXML(
            model_name=model_name,
            conf_threshold=conf,
            kpt_threshold=kpt_conf,
            device=device
        )
        
        if category == 'normal':
            # XML bbox 기반 추출 (ID가 있는 모든 인물) - XML이 있는 경우에만
            if xml_path_or_none is not None:
                success = extractor.extract_with_xml(video_path, Path(xml_path_or_none), str(output_json_path))
            else:
                # XML 없으면 단순 추출 (정상 데이터지만 XML이 없거나 제외된 경우)
                success = extractor.extract_simple(video_path, str(output_json_path))
        else:
            # 단순 1인 추출 (이상 데이터)
            success = extractor.extract_simple(video_path, str(output_json_path))
        
        if success:
            return ('success', filename, category)
        else:
            return ('fail_extract', filename, category)
    except Exception as e:
        print(f"[ERROR] Exception processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return ('fail_exception', filename, category)


def main():
    parser = argparse.ArgumentParser(description='Extract skeleton with XML bbox (normal) or simple mode (abnormal)')
    parser.add_argument('--model', type=str, default='yolo11x-pose.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--kpt_conf', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--data_type', type=str, default='all', choices=['train', 'mlp_train', 'test', 'all'])
    parser.add_argument('--num_workers', type=int, default=12)
    
    args = parser.parse_args()
    
    # Windows 멀티프로세싱
    if os.name == 'nt':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    # DIR 입력: data_split 제거 이후에는 skeleton_extracted 기반 DIR을 사용하세요.
    # 예) base_dir / "skeleton_extracted" / "test" / "skeleton_data"
    data_split_dir = None  # deprecated
    
    # 정상 데이터 경로
    normal_video_root = Path("D:/AI-HUB_shoping/shoping_data/Training/raw_data")
    normal_xml_dirs = [
        Path("D:/AI-HUB_shoping/shoping_data/Training/label_data/TL_01.매장이동_01.매장이동"),
    ]
    
    # XML을 사용하지 않을 디렉토리 (단순 추출)
    excluded_xml_dirs = [
        Path("D:/AI-HUB_shoping/shoping_data/Training/label_data/TL_02.구매행동_03.시험"),
        Path("D:/AI-HUB_shoping/shoping_data/Training/label_data/TL_02.구매행동_04.구매"),
        Path("D:/AI-HUB_shoping/shoping_data/Training/label_data/TL_02.구매행동_05.반품"),
        Path("D:/AI-HUB_shoping/shoping_data/Training/label_data/TL_02.구매행동_06.비교"),
        Path("D:/AI-HUB_shoping/shoping_data/Training/label_data/TL_02.구매행동_02.선택"),
    ]
    
    # 이상 데이터 경로
    abnormal_video_root = Path("D:/AI-HUB_shoplifting/shoplift_data/Training/raw_data/Shoplift")
    
    # XML 통합 검색 함수
    def find_xml_for_video(xml_filename, xml_dirs):
        """
        여러 XML 폴더에서 파일 검색
        """
        for xml_dir in xml_dirs:
            xml_path = xml_dir / xml_filename
            if xml_path.exists():
                return xml_path
        
        # 못 찾은 경우 시도한 경로들 출력
        print(f"[DEBUG] XML not found: {xml_filename}")
        print(f"[DEBUG] Searched in:")
        for xml_dir in xml_dirs:
            print(f"  - {xml_dir / xml_filename}")
        
        return None
    
    # 처리할 데이터
    datasets_to_process = []
    if args.data_type in ['train', 'all']:
        datasets_to_process.append(('train_stgnf', 'train_data.json', 'train_stgnf'))
    if args.data_type in ['mlp_train', 'all']:
        datasets_to_process.append(('train_attention', 'attention_train_data.json', 'train_attention'))
    if args.data_type in ['test', 'all']:
        datasets_to_process.append(('test', 'test_data.json', 'test'))
    
    print(f"\n[INFO] Model: {args.model}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Workers: {args.num_workers}")
    print(f"[INFO] Mode: XML bbox (normal - all IDs) + Simple (abnormal)")
    
    for dataset_name, json_filename, skeleton_dir_name in datasets_to_process:
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # DIR 입력 사용: 외부 JSON 없이 DIR을 직접 스캔하도록 변경 필요
        json_path = None  # deprecated
        if not json_path.exists():
            print(f"[WARNING] JSON not found: {json_path}")
            continue
        
        output_skeleton_dir = base_dir / "skeleton_extracted" / skeleton_dir_name / "skeleton_data"
        output_skeleton_dir.mkdir(parents=True, exist_ok=True)
        
        videos = load_json_split(json_path)
        print(f"[INFO] Total videos: {len(videos)}")
        
        # 멀티프로세싱 인자 준비 (C_2 파일과 특정 디렉토리는 단순 추출)
        process_args = []
        for video_info in videos:
            filename = video_info['filename']
            
            # C_2로 시작하는 파일은 XML 없이 단순 추출
            if filename.startswith('C_2'):
                print(f"[INFO] C_2 file detected, using simple extraction: {filename}")
                process_args.append((
                    video_info, output_skeleton_dir, args.model, args.conf, args.kpt_conf,
                    args.device, None  # XML 없이 처리
                ))
                continue
            
            if video_info['category'] == 'normal':
                # XML 파일 검색
                xml_filename = filename.replace('.mp4', '.xml')
                xml_path = find_xml_for_video(xml_filename, normal_xml_dirs)
                
                # XML을 찾았지만 제외 디렉토리에 있는 경우 단순 추출
                if xml_path is not None:
                    is_excluded = any(str(xml_path).startswith(str(excluded_dir)) for excluded_dir in excluded_xml_dirs)
                    if is_excluded:
                        print(f"[INFO] Excluded XML dir detected, using simple extraction: {filename}")
                        process_args.append((
                            video_info, output_skeleton_dir, args.model, args.conf, args.kpt_conf,
                            args.device, None  # XML 없이 처리
                        ))
                        continue
                
                if xml_path is None:
                    print(f"[WARNING] XML not found for {filename}, using simple extraction")
                    # XML 없으면 단순 추출로 처리
                    process_args.append((
                        video_info, output_skeleton_dir, args.model, args.conf, args.kpt_conf,
                        args.device, None
                    ))
                    continue
                
                # XML 전체 경로 전달
                process_args.append((
                    video_info, output_skeleton_dir, args.model, args.conf, args.kpt_conf,
                    args.device, str(xml_path)
                ))
            else:
                # 이상 데이터는 XML 불필요 (None 전달)
                process_args.append((
                    video_info, output_skeleton_dir, args.model, args.conf, args.kpt_conf,
                    args.device, None
                ))
        
        success_count = 0
        fail_video_count = 0
        fail_xml_count = 0
        fail_extract_count = 0
        fail_exception_count = 0
        skip_count = 0
        
        print(f"[INFO] Starting extraction...")

        
        with mp.Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_video, process_args),
                total=len(process_args),
                desc=f"Extracting {dataset_name}"
            ))
        
        for status, filename, category in results:
            if status == 'success':
                success_count += 1
            elif status == 'skip':
                skip_count += 1
            elif status == 'fail_video':
                fail_video_count += 1
            elif status == 'fail_xml':
                fail_xml_count += 1
            elif status == 'fail_extract':
                fail_extract_count += 1
            elif status == 'fail_exception':
                fail_exception_count += 1
        
        print(f"\n[RESULT] {dataset_name.upper()}")
        print(f"  Success: {success_count}")
        print(f"  Skipped (already exists): {skip_count}")
        print(f"  Failed (video not found): {fail_video_count}")
        print(f"  Failed (XML not found): {fail_xml_count}")
        print(f"  Failed (extraction error): {fail_extract_count}")
        print(f"  Failed (exception): {fail_exception_count}")
        print(f"  Total processed: {success_count + skip_count}")
        print(f"  Output dir: {output_skeleton_dir}")
    
    print(f"\n{'='*80}")
    print("All done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()