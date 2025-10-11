import os
import xml.etree.ElementTree as ET
import json
import random
import numpy as np
from pathlib import Path
import cv2
from collections import defaultdict


class DataSplitter:
    def __init__(self):
        # 데이터 경로 설정
        self.normal_data_path = "D:/AI-HUB_shoping/shoping_data/Training/raw_data"
        self.abnormal_data_path = "D:/AI-HUB_shoplifting/shoplift_data/Training/raw_data/Shoplift"
        self.abnormal_label_path = "D:/AI-HUB_shoplifting/shoplift_data/Training/label_data/Shoplift"
        
        # 출력 경로 설정 (JSON 파일들)
        self.output_base_path = "./output"
        self.train_json = os.path.join(self.output_base_path, "train_data.json")
        self.mlp_train_json = os.path.join(self.output_base_path, "mlp_train_data.json")
        self.test_json = os.path.join(self.output_base_path, "test_data.json")
        self.gt_path = os.path.join(self.output_base_path, "gt")
        
        # GT 하위 폴더
        self.mlp_train_gt_path = os.path.join(self.gt_path, "mlp_train_gt")
        self.test_gt_path = os.path.join(self.gt_path, "test_gt")
        
    def create_output_directories(self):
        """출력 디렉토리 생성"""
        directories = [
            self.output_base_path,
            self.gt_path, self.mlp_train_gt_path, self.test_gt_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def parse_xml_for_theft_frames(self, xml_path):
        """XML 파일에서 theft_start, theft_end 프레임 정보 추출"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            theft_start_frame = None
            theft_end_frame = None
            total_frames = 0
            
            # 전체 프레임 수 확인
            task = root.find('meta/task')
            if task is not None:
                stop_frame = task.find('stop_frame')
                if stop_frame is not None and stop_frame.text is not None:
                    total_frames = int(stop_frame.text) + 1  # 0부터 시작하므로 +1
            
            # theft_start와 theft_end 트랙 찾기
            tracks = root.findall('track')
            for track in tracks:
                label = track.get('label')
                if label == 'theft_start':
                    # 첫 번째 box의 frame 찾기
                    box = track.find('box[@outside="0"]')  # outside="0"인 첫 번째 박스
                    if box is not None:
                        frame_attr = box.get('frame')
                        if frame_attr is not None:
                            theft_start_frame = int(frame_attr)
                        
                elif label == 'theft_end':
                    # 첫 번째 box의 frame 찾기
                    box = track.find('box[@outside="0"]')  # outside="0"인 첫 번째 박스
                    if box is not None:
                        frame_attr = box.get('frame')
                        if frame_attr is not None:
                            theft_end_frame = int(frame_attr)
            
            return theft_start_frame, theft_end_frame, total_frames
            
        except Exception as e:
            print(f"XML 파싱 오류 {xml_path}: {e}")
            return None, None, 0
    
    def get_video_frame_count(self, video_path):
        """비디오 파일의 프레임 수 확인"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count
        except:
            return 0
    
    def create_ground_truth_labels(self, video_name, theft_start, theft_end, total_frames, is_normal=False):
        """Ground truth 라벨 생성 (0: 정상, 1: 이상)"""
        if is_normal:
            # 정상 데이터는 모든 프레임이 0
            return np.zeros(total_frames, dtype=np.int32)
        else:
            # 이상 데이터는 theft_start부터 theft_end까지 1, 나머지는 0
            labels = np.zeros(total_frames, dtype=np.int32)
            if theft_start is not None and theft_end is not None:
                labels[theft_start:theft_end+1] = 1
            return labels
    
    def get_all_files(self):
        """모든 정상/이상 데이터 파일 목록 수집"""
        normal_files = []
        abnormal_files = []
        abnormal_annotations = {}
        
        # 정상 데이터 수집 - 6개 카테고리에서 균등하게
        if os.path.exists(self.normal_data_path):
            categories = []
            category_files = {}
            
            # 각 카테고리별 파일 수집
            for category_dir in os.listdir(self.normal_data_path):
                category_path = os.path.join(self.normal_data_path, category_dir)
                if os.path.isdir(category_path):
                    category_files_list = []
                    # 하위 폴더까지 재귀적으로 검색
                    for root, dirs, files in os.walk(category_path):
                        for file in files:
                            if file.endswith('.mp4'):
                                # 상대 경로로 저장 (카테고리 포함)
                                rel_path = os.path.relpath(os.path.join(root, file), self.normal_data_path)
                                category_files_list.append(rel_path)
                    
                    if category_files_list:
                        categories.append(category_dir)
                        category_files[category_dir] = category_files_list
                        print(f"카테고리 {category_dir}: {len(category_files_list)}개 파일")
            
            # 각 카테고리에서 파일들을 셔플
            for category in categories:
                random.shuffle(category_files[category])
            
            self.category_files = category_files
            self.categories = categories
            
            # 전체 정상 파일 리스트 생성 (나중에 균등 분할용)
            for category in categories:
                normal_files.extend(category_files[category])
        
        # 이상 데이터 수집 및 annotation 정보 추출
        if os.path.exists(self.abnormal_data_path):
            for file in os.listdir(self.abnormal_data_path):
                if file.endswith('.mp4'):
                    abnormal_files.append(file)
                    
                    # 해당하는 XML 파일 찾기
                    xml_file = file.replace('.mp4', '.xml')
                    xml_path = os.path.join(self.abnormal_label_path, xml_file)
                    
                    if os.path.exists(xml_path):
                        theft_start, theft_end, total_frames = self.parse_xml_for_theft_frames(xml_path)
                        abnormal_annotations[file] = {
                            'theft_start': theft_start,
                            'theft_end': theft_end,
                            'total_frames': total_frames
                        }
                    else:
                        print(f"XML 파일을 찾을 수 없음: {xml_path}")
        
        return normal_files, abnormal_files, abnormal_annotations
    
    def distribute_normal_files_evenly(self, total_needed):
        """6개 카테고리에서 균등하게 파일 선택"""
        if not hasattr(self, 'category_files') or not hasattr(self, 'categories'):
            return []
        
        # 반품 카테고리 특별 처리 (41개씩 각 폴더에 배분)
        special_category = "TS_02.구매행동_05.반품_2"
        special_allocation = 41
        
        files_per_category = total_needed // len(self.categories)
        remaining_files = total_needed % len(self.categories)
        
        selected_files = []
        category_usage = {}  # 각 카테고리별 사용된 파일 수 추적
        
        for i, category in enumerate(self.categories):
            # 반품 카테고리는 항상 41개씩 할당
            if category == special_category:
                needed_from_category = special_allocation
            else:
                # 기본 할당량 + 나머지 분배 (반품 제외한 나머지 카테고리에 분배)
                remaining_categories = len(self.categories) - 1  # 반품 제외
                adjusted_total = total_needed - special_allocation  # 반품 할당분 제외
                needed_from_category = adjusted_total // remaining_categories
                if i < (adjusted_total % remaining_categories):
                    needed_from_category += 1
            
            # 이미 사용된 파일 수 확인
            used_count = category_usage.get(category, 0)
            available_files = self.category_files[category][used_count:]
            
            # 필요한 만큼 선택
            selected_count = min(needed_from_category, len(available_files))
            selected = available_files[:selected_count]
            
            selected_files.extend(selected)
            category_usage[category] = used_count + selected_count
            
            print(f"카테고리 {category}: {selected_count}개 선택 (총 사용: {category_usage[category]}개)")
        
        # 전역 사용량 업데이트
        if not hasattr(self, 'global_category_usage'):
            self.global_category_usage = {}
        
        for category in category_usage:
            if category not in self.global_category_usage:
                self.global_category_usage[category] = 0
            self.global_category_usage[category] += category_usage[category]
            # 사용된 파일들을 리스트에서 제거하여 중복 방지
            self.category_files[category] = self.category_files[category][category_usage[category]:]
        
        return selected_files
    
    def split_data(self):
        """데이터를 train, mlp_train, test로 분할"""
        normal_files, abnormal_files, abnormal_annotations = self.get_all_files()
        
        print(f"정상 데이터: {len(normal_files)}개")
        print(f"이상 데이터: {len(abnormal_files)}개")
        
        # 모든 이상 데이터 사용
        total_abnormal = len(abnormal_files)
        
        # mlp_train: 정상 5 : 이상 5
        # test: 정상 6 : 이상 4
        # 이상 데이터를 mlp_train과 test로 나누기 (5:4 비율)
        mlp_abnormal_count = int(total_abnormal * 5 / 9)
        test_abnormal_count = total_abnormal - mlp_abnormal_count
        
        # 정상 데이터 개수 계산
        mlp_normal_count = mlp_abnormal_count  # 5:5 비율
        test_normal_count = int(test_abnormal_count * 6 / 4)  # 6:4 비율
        train_normal_count = mlp_normal_count + test_normal_count  # 요구사항: train = mlp_train + test 개수
        
        print(f"\n분할 계획:")
        print(f"MLP Train - 정상: {mlp_normal_count}, 이상: {mlp_abnormal_count}")
        print(f"Test - 정상: {test_normal_count}, 이상: {test_abnormal_count}")
        print(f"Train - 정상: {train_normal_count}")
        
        # 이상 데이터 셔플 및 분할 (중복 방지)
        random.shuffle(abnormal_files)
        mlp_abnormal = abnormal_files[:mlp_abnormal_count]
        test_abnormal = abnormal_files[mlp_abnormal_count:mlp_abnormal_count + test_abnormal_count]
        
        # 이상 데이터 개수 확인
        print(f"실제 이상 데이터 분할:")
        print(f"MLP Train 이상: {len(mlp_abnormal)}개")
        print(f"Test 이상: {len(test_abnormal)}개")
        print(f"총 이상 데이터 사용: {len(mlp_abnormal) + len(test_abnormal)}개 / {total_abnormal}개")
        
        # 정상 데이터 균등 분배
        print(f"\n정상 데이터 균등 분배:")
        print("=== MLP Train용 정상 데이터 ===")
        mlp_normal = self.distribute_normal_files_evenly(mlp_normal_count)
        
        print("\n=== Test용 정상 데이터 ===")
        test_normal = self.distribute_normal_files_evenly(test_normal_count)
        
        print("\n=== Train용 정상 데이터 ===")
        train_normal = self.distribute_normal_files_evenly(train_normal_count)
        
        return {
            'mlp_train': {'normal': mlp_normal, 'abnormal': mlp_abnormal},
            'test': {'normal': test_normal, 'abnormal': test_abnormal},
            'train': {'normal': train_normal, 'abnormal': []},
            'annotations': abnormal_annotations
        }
    
    def save_data_splits_and_create_gt(self, data_split):
        """데이터 분할 정보를 JSON으로 저장하고 GT 파일 생성"""
        annotations = data_split['annotations']
        
        # JSON 데이터 구조 생성
        json_data = {}
        
        # MLP Train 데이터 처리
        print("\nMLP Train 데이터 JSON 생성 및 GT 생성 중...")
        mlp_train_data = {
            'normal': [],
            'abnormal': []
        }
        
        # 정상 데이터 처리
        for file_path in data_split['mlp_train']['normal']:
            src = os.path.join(self.normal_data_path, file_path)
            filename = os.path.basename(file_path)
            
            if os.path.exists(src):
                # JSON에 파일 정보 추가
                mlp_train_data['normal'].append({
                    'filename': filename,
                    'full_path': src,
                    'relative_path': file_path,
                    'label': 0  # 정상 데이터
                })
                
                # GT 파일 생성 (정상 데이터)
                frame_count = self.get_video_frame_count(src)
                gt_labels = self.create_ground_truth_labels(filename, None, None, frame_count, is_normal=True)
                gt_file = os.path.join(self.mlp_train_gt_path, filename.replace('.mp4', '.npy'))
                np.save(gt_file, gt_labels)
        
        # 이상 데이터 처리
        for file in data_split['mlp_train']['abnormal']:
            src = os.path.join(self.abnormal_data_path, file)
            
            if os.path.exists(src):
                # 어노테이션 정보 가져오기
                ann_info = annotations.get(file, {})
                
                # JSON에 파일 정보 추가
                mlp_train_data['abnormal'].append({
                    'filename': file,
                    'full_path': src,
                    'label': 1,  # 이상 데이터
                    'theft_start': ann_info.get('theft_start'),
                    'theft_end': ann_info.get('theft_end'),
                    'total_frames': ann_info.get('total_frames', 0)
                })
                
                # GT 파일 생성 (이상 데이터)
                if file in annotations:
                    ann = annotations[file]
                    frame_count = self.get_video_frame_count(src)
                    gt_labels = self.create_ground_truth_labels(
                        file, ann['theft_start'], ann['theft_end'], frame_count, is_normal=False
                    )
                    gt_file = os.path.join(self.mlp_train_gt_path, file.replace('.mp4', '.npy'))
                    np.save(gt_file, gt_labels)
        
        # MLP Train JSON 저장
        with open(self.mlp_train_json, 'w', encoding='utf-8') as f:
            json.dump(mlp_train_data, f, ensure_ascii=False, indent=2)
        
        # Test 데이터 처리
        print("Test 데이터 JSON 생성 및 GT 생성 중...")
        test_data = {
            'normal': [],
            'abnormal': []
        }
        
        # 정상 데이터 처리
        for file_path in data_split['test']['normal']:
            src = os.path.join(self.normal_data_path, file_path)
            filename = os.path.basename(file_path)
            
            if os.path.exists(src):
                # JSON에 파일 정보 추가
                test_data['normal'].append({
                    'filename': filename,
                    'full_path': src,
                    'relative_path': file_path,
                    'label': 0  # 정상 데이터
                })
                
                # GT 파일 생성 (정상 데이터)
                frame_count = self.get_video_frame_count(src)
                gt_labels = self.create_ground_truth_labels(filename, None, None, frame_count, is_normal=True)
                gt_file = os.path.join(self.test_gt_path, filename.replace('.mp4', '.npy'))
                np.save(gt_file, gt_labels)
        
        # 이상 데이터 처리
        for file in data_split['test']['abnormal']:
            src = os.path.join(self.abnormal_data_path, file)
            
            if os.path.exists(src):
                # 어노테이션 정보 가져오기
                ann_info = annotations.get(file, {})
                
                # JSON에 파일 정보 추가
                test_data['abnormal'].append({
                    'filename': file,
                    'full_path': src,
                    'label': 1,  # 이상 데이터
                    'theft_start': ann_info.get('theft_start'),
                    'theft_end': ann_info.get('theft_end'),
                    'total_frames': ann_info.get('total_frames', 0)
                })
                
                # GT 파일 생성 (이상 데이터)
                if file in annotations:
                    ann = annotations[file]
                    frame_count = self.get_video_frame_count(src)
                    gt_labels = self.create_ground_truth_labels(
                        file, ann['theft_start'], ann['theft_end'], frame_count, is_normal=False
                    )
                    gt_file = os.path.join(self.test_gt_path, file.replace('.mp4', '.npy'))
                    np.save(gt_file, gt_labels)
        
        # Test JSON 저장
        with open(self.test_json, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # Train 데이터 처리 (정상 데이터만)
        print("Train 데이터 JSON 생성 중...")
        train_data = {
            'normal': [],
            'abnormal': []
        }
        
        # 정상 데이터 처리
        for file_path in data_split['train']['normal']:
            src = os.path.join(self.normal_data_path, file_path)
            filename = os.path.basename(file_path)
            
            if os.path.exists(src):
                # JSON에 파일 정보 추가
                train_data['normal'].append({
                    'filename': filename,
                    'full_path': src,
                    'relative_path': file_path,
                    'label': 0  # 정상 데이터
                })
        
        # Train JSON 저장
        with open(self.train_json, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        # 요약 정보 출력
        print(f"\n=== 데이터 분할 결과 ===")
        print(f"MLP Train: 정상 {len(mlp_train_data['normal'])}개, 이상 {len(mlp_train_data['abnormal'])}개")
        print(f"Test: 정상 {len(test_data['normal'])}개, 이상 {len(test_data['abnormal'])}개")
        print(f"Train: 정상 {len(train_data['normal'])}개")
        print(f"\nJSON 파일 저장 위치:")
        print(f"- {self.mlp_train_json}")
        print(f"- {self.test_json}")
        print(f"- {self.train_json}")
        print(f"\nGT 파일 저장 위치:")
        print(f"- {self.mlp_train_gt_path}")
        print(f"- {self.test_gt_path}")
    
    def run(self):
        """전체 데이터 분할 프로세스 실행"""
        print("데이터 분할 프로세스 시작...")
        
        # 출력 디렉토리 생성
        self.create_output_directories()
        
        # 데이터 분할
        data_split = self.split_data()
        
        # JSON 파일 저장 및 GT 생성
        self.save_data_splits_and_create_gt(data_split)
        
        print("\n데이터 분할 완료!")
        print(f"결과는 {self.output_base_path} 폴더에 저장되었습니다.")
        print("- JSON 파일: 각 데이터셋의 파일 경로와 메타데이터")
        print("- GT 폴더: Ground Truth 라벨 파일들 (.npy 형식)")


if __name__ == "__main__":
    # 랜덤 시드 설정
    random.seed(42)
    
    # 데이터 분할 실행
    splitter = DataSplitter()
    splitter.run()