"""
전체 데이터셋 처리기
PoseLift 논문 방식으로 훈련/테스트 데이터셋 생성

데이터셋 구성:
- 훈련: 정상 데이터만 (AI-HUB_shoping Training + Validation)
- 테스트: 정상 데이터 일부 + 이상 데이터 전체 (겹치지 않게)

저장 형식:
- PoseLift JSON 형식
- NPY 레이블 파일
"""

import os
import json
import numpy as np
from pathlib import Path
import random
from keypoint_extract import PoseLiftDatasetProcessor
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time

def process_single_video_parallel(video_path):
    """병렬처리용 단일 비디오 처리 함수"""
    try:
        processor = PoseLiftDatasetProcessor()
        pose_data, labels = processor.process_single_video(video_path)
        if pose_data:
            return video_path, pose_data, labels, True
        else:
            return video_path, None, None, False
    except Exception as e:
        print(f"오류 ({os.path.basename(video_path)}): {e}")
        return video_path, None, None, False

def process_single_video_with_xml(video_path, xml_path=None):
    """병렬처리용 XML 포함 비디오 처리 함수"""
    try:
        processor = PoseLiftDatasetProcessor()
        pose_data, labels = processor.process_single_video(video_path, xml_path)
        if pose_data:
            return video_path, pose_data, labels, True
        else:
            return video_path, None, None, False
    except Exception as e:
        print(f"오류 ({os.path.basename(video_path)}): {e}")
        return video_path, None, None, False

class DatasetProcessor:
    def __init__(self):
        self.processor = PoseLiftDatasetProcessor()
        # 16코어 CPU 최적화: 시스템 안정성을 위해 50% 사용 (8개 프로세스)
        self.max_workers = min(cpu_count() // 2, 8)  # 16코어의 50% = 8개 프로세스
        
        # 데이터 경로 설정
        self.normal_paths = [
            r"D:\AI-HUB_shoping\shoping_data\Training",
            r"D:\AI-HUB_shoping\shoping_data\Validation"
        ]
        
        self.abnormal_paths = [
            r"D:\AI-HUB_shoplifting\shoplift_data\Training", 
            r"D:\AI-HUB_shoplifting\shoplift_data\Validation"
        ]
        
        # 출력 디렉토리
        self.output_dir = Path("dataset_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # 데이터 분할 비율 설정
        self.train_test_ratio = 4  # Train:Test = 4:1
        self.normal_abnormal_ratio = 0.7  # 정상:이상 = 7:3 (0.7 = 70% 정상, 30% 이상)
        # 다른 비율 옵션: 0.6 (정상:이상 = 6:4)
        
    def find_video_files(self, base_path):
        """지정된 경로에서 모든 비디오 파일 찾기"""
        video_files = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(root, file)
                    video_files.append(video_path)
        
        return video_files
    
    def find_xml_file(self, video_path):
        """비디오 파일에 대응하는 XML 파일 찾기"""
        # 비디오 파일명에서 확장자 제거
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # XML 파일이 있을 수 있는 경로들 탐색
        video_dir = os.path.dirname(video_path)
        parent_dir = os.path.dirname(video_dir)
        
        # 가능한 XML 경로들
        possible_xml_paths = [
            os.path.join(video_dir, f"{base_name}.xml"),
            os.path.join(parent_dir, "label_data", os.path.basename(video_dir), f"{base_name}.xml"),
            os.path.join(os.path.dirname(parent_dir), "label_data", os.path.basename(video_dir), f"{base_name}.xml")
        ]
        
        for xml_path in possible_xml_paths:
            if os.path.exists(xml_path):
                return xml_path
        
        return None
    
    def process_normal_dataset(self):
        """정상 데이터 전체 처리 (카테고리별로 모두 수집)"""
        print("=== 정상 데이터셋 수집 중 ===")
        
        all_normal_videos = []
        for path in self.normal_paths:
            if os.path.exists(path):
                videos = self.find_video_files(path)
                all_normal_videos.extend(videos)
                print(f"경로 {path}: {len(videos)}개 비디오 발견")
        
        print(f"총 정상 비디오: {len(all_normal_videos)}개")
        return all_normal_videos
    
    def group_videos_by_category(self, video_paths):
        """비디오를 카테고리별로 그룹화 (AI-HUB 실제 구조 기준)"""
        categories = {}
        
        for video_path in video_paths:
            # 경로에서 카테고리 추출
            path_parts = video_path.replace('\\', '/').split('/')
            category = 'unknown'
            
            # AI-HUB 폴더명 패턴: TS_번호.행동명_세부번호.세부행동_번호
            for part in path_parts:
                if 'TS_' in part and '.' in part:
                    # TS_01.매장이동_01.매장이동_1 -> 매장이동
                    # TS_02.구매행동_02.선택_1 -> 구매행동
                    try:
                        # 첫 번째 점 이후, 두 번째 언더스코어 이전까지가 주요 행동
                        parts = part.split('.')
                        if len(parts) >= 2:
                            main_action = parts[1].split('_')[0]  # 구매행동, 매장이동 등
                            category = main_action
                            break
                    except:
                        continue
            
            # 세부 분류가 필요한 경우 더 자세히 분류
            if category != 'unknown':
                for part in path_parts:
                    if 'TS_' in part and category in part:
                        # TS_02.구매행동_02.선택_1 -> 구매행동_선택
                        # TS_02.구매행동_06.비교_1 -> 구매행동_비교  
                        try:
                            parts = part.split('.')
                            if len(parts) >= 3:
                                sub_action = parts[2].split('_')[0]  # 선택, 비교, 구매 등
                                category = f"{category}_{sub_action}"
                                break
                        except:
                            continue
            
            # 디버깅을 위한 unknown 경로 출력 (처음 3개만)
            if category == 'unknown' and len(categories.get('unknown', [])) < 3:
                print(f"[DEBUG] 카테고리 인식 실패: {video_path}")
                print(f"[DEBUG] 경로 구성 요소: {path_parts}")
            
            if category not in categories:
                categories[category] = []
            categories[category].append(video_path)
        
        return categories

    def calculate_optimal_ratios(self, normal_videos, abnormal_videos):
        """이상 탐지용 비율 계산 및 데이터 분할 (카테고리 균등 분포)"""
        print("\n=== 이상 탐지 데이터셋 비율 계산 중 ===")
        
        # 1. 정상 데이터를 카테고리별로 그룹화
        normal_categories = self.group_videos_by_category(normal_videos)
        print(f"정상 데이터 카테고리: {list(normal_categories.keys())}")
        for category, videos in normal_categories.items():
            print(f"  - {category}: {len(videos)}개")
        
        # 2. 학습 최적화를 위한 데이터 개수 제한
        max_abnormal_count = min(len(abnormal_videos), 50)  # 이상 데이터 최대 50개로 제한
        max_test_normal_count = int(max_abnormal_count * (self.normal_abnormal_ratio / (1 - self.normal_abnormal_ratio)))  # 7:3 비율
        
        # Train:Test = 4:1 비율 적용
        test_total = max_test_normal_count + max_abnormal_count
        train_normal_count = int(test_total * self.train_test_ratio)  # Train = Test * 4
        
        # 실제 사용할 개수
        test_abnormal_count = max_abnormal_count
        test_normal_count = max_test_normal_count
        
        print(f"\n목표 데이터 개수 (학습 시간 최적화):")
        print(f"  - Train: {train_normal_count:.0f}개 (정상만)")
        print(f"  - Test: {test_total}개 (정상:{test_normal_count}, 이상:{test_abnormal_count})")
        
        # 3. 각 카테고리에서 비율에 맞게 데이터 추출
        train_normal_videos = []
        test_normal_videos = []
        
        for category, videos in normal_categories.items():
            random.shuffle(videos)
            
            # 이 카테고리에서 필요한 개수 계산
            category_ratio = len(videos) / len(normal_videos)
            category_train_count = int(train_normal_count * category_ratio)
            category_test_count = int(test_normal_count * category_ratio)
            
            # 카테고리별 최대 사용 가능 개수 확인
            available_count = len(videos)
            total_needed = category_train_count + category_test_count
            
            if total_needed > available_count:
                # 비율 유지하면서 사용 가능한 범위 내에서 조정
                ratio = available_count / total_needed
                category_train_count = int(category_train_count * ratio)
                category_test_count = available_count - category_train_count
            
            # 카테고리별 데이터 분할
            train_normal_videos.extend(videos[:category_train_count])
            test_normal_videos.extend(videos[category_train_count:category_train_count + category_test_count])
            
            print(f"  - {category} 분할: Train {category_train_count}개, Test {category_test_count}개")
        
        # 4. 이상 데이터는 테스트에만 사용 (개수 제한)
        random.shuffle(abnormal_videos)
        test_abnormal_videos = abnormal_videos[:test_abnormal_count]  # 제한된 개수만 사용
        train_abnormal_videos = []  # 훈련에는 이상 데이터 없음!
        
        # 최종 결과
        actual_test_normal_ratio = len(test_normal_videos) / (len(test_normal_videos) + len(test_abnormal_videos))
        actual_test_abnormal_ratio = len(test_abnormal_videos) / (len(test_normal_videos) + len(test_abnormal_videos))
        actual_train_test_ratio = len(train_normal_videos) / (len(test_normal_videos) + len(test_abnormal_videos))
        
        print(f"\n최종 데이터 분할 결과:")
        print(f"Train 데이터: {len(train_normal_videos)}개 (정상만)")
        print(f"Test 데이터: {len(test_normal_videos) + len(test_abnormal_videos)}개")
        print(f"  - 정상: {len(test_normal_videos)}개 ({actual_test_normal_ratio:.1%})")
        print(f"  - 이상: {len(test_abnormal_videos)}개 ({actual_test_abnormal_ratio:.1%})")
        print(f"Train:Test 비율 = {actual_train_test_ratio:.1f}:1")
        
        return train_normal_videos, test_normal_videos, train_abnormal_videos, test_abnormal_videos
    
    def process_abnormal_dataset(self):
        """이상 데이터 전체 처리"""
        print("\n=== 이상 데이터셋 수집 중 ===")
        
        all_abnormal_videos = []
        for path in self.abnormal_paths:
            if os.path.exists(path):
                videos = self.find_video_files(path)
                all_abnormal_videos.extend(videos)
                print(f"경로 {path}: {len(videos)}개 비디오 발견")
        
        print(f"총 이상 비디오: {len(all_abnormal_videos)}개")
        return all_abnormal_videos
    
    def generate_meaningful_video_name(self, video_path, is_abnormal, index):
        """의미있는 비디오 이름 생성"""
        # 비디오 파일명에서 확장자 제거
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 경로에서 카테고리 정보 추출
        path_parts = video_path.replace('\\', '/').split('/')
        
        if is_abnormal:
            # 이상 행동 (shoplifting)
            prefix = "abnormal_shoplifting"
        else:
            # 정상 행동 (shopping)
            prefix = "normal_shopping"
        
        # 3자리 인덱스로 포맷
        return f"{prefix}_{index:03d}_{base_name}"
    
    def process_video_batch(self, video_list, output_prefix, is_abnormal=False):
        """병렬처리로 비디오 배치 처리"""
        print(f"\n{output_prefix} 처리 시작... ({len(video_list)}개 비디오)")
        print(f"병렬처리 시작 ({self.max_workers}개 프로세스)")
        
        all_pose_data = {}
        all_labels = []
        processed_count = 0
        failed_count = 0
        video_metadata = {}  # 비디오 메타데이터 저장
        
        start_time = time.time()
        
        # 병렬처리로 비디오 처리
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 비디오에 대해 작업 제출 (XML 경로 포함)
            future_to_video = {}
            for video_path in video_list:
                xml_path = None
                if is_abnormal:
                    xml_path = self.find_xml_file(video_path)
                    if xml_path is None:
                        print(f"XML 파일을 찾을 수 없음: {os.path.basename(video_path)}")
                        failed_count += 1
                        continue
                
                future = executor.submit(process_single_video_with_xml, video_path, xml_path)
                future_to_video[future] = video_path
            
            # 완료된 작업들 처리
            for i, future in enumerate(as_completed(future_to_video), 1):
                video_path = future_to_video[future]
                try:
                    video_path, pose_data, labels, success = future.result()
                    
                    # 의미있는 비디오 이름 생성
                    meaningful_name = self.generate_meaningful_video_name(video_path, is_abnormal, processed_count + 1)
                    
                    print(f"[{i}/{len(future_to_video)}] {meaningful_name} - {'성공' if success else '실패'}")
                    
                    if success and pose_data:
                        # 의미있는 ID로 person_id 재생성
                        for person_id, frames in pose_data.items():
                            meaningful_id = f"{meaningful_name}_person_{person_id}"
                            all_pose_data[meaningful_id] = frames
                        
                        # 비디오 메타데이터 저장 (라벨 매칭 정보 포함)
                        frame_count = len(next(iter(pose_data.values())))
                        video_metadata[meaningful_name] = {
                            'original_path': video_path,
                            'category': 'abnormal' if is_abnormal else 'normal',
                            'frame_count': frame_count,
                            'person_count': len(pose_data),
                            'video_index': processed_count + 1,  # 고유 비디오 번호
                            'label_start_idx': len(all_labels),  # GT 파일에서 이 비디오 라벨 시작 인덱스
                            'label_end_idx': len(all_labels) + frame_count - 1  # GT 파일에서 이 비디오 라벨 끝 인덱스
                        }
                        
                        # 레이블 추가
                        if labels is not None and len(labels) > 0:
                            all_labels.extend(labels)
                        else:
                            # 정상 데이터의 경우 모든 프레임을 0으로 설정
                            frame_count = len(next(iter(pose_data.values())))
                            all_labels.extend([0] * frame_count)
                        
                        processed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    print(f"처리 중 오류: {e}")
                    failed_count += 1
        
        elapsed_time = time.time() - start_time
        
        print(f"\n=== {output_prefix} 처리 완료 ===")
        print(f"성공: {processed_count}개, 실패: {failed_count}개")
        print(f"총 사람: {len(all_pose_data)}명")
        print(f"총 프레임: {len(all_labels)}개")
        
        return all_pose_data, np.array(all_labels), video_metadata
    
    def save_dataset(self, pose_data, labels, prefix, save_gt=False, metadata=None):
        """데이터셋을 비디오별 개별 파일로 저장"""
        # split별 폴더 생성 (train, test)
        split_dir = self.output_dir / prefix
        split_dir.mkdir(exist_ok=True)
        
        if save_gt:
            # GT 폴더 생성
            gt_dir = split_dir / "ground_truth"
            gt_dir.mkdir(exist_ok=True)
        
        saved_files = []
        video_count = 0
        
        # 비디오별로 개별 파일 저장
        if metadata:
            for video_name, video_info in metadata.items():
                # 1. 해당 비디오의 포즈 데이터 추출
                video_pose_data = {}
                for person_id, frames in pose_data.items():
                    if person_id.startswith(video_name):
                        video_pose_data[person_id] = frames
                
                # 2. 비디오별 JSON 파일 저장
                video_json_path = split_dir / f"{video_name}.json"
                self.processor.save_results_to_json(video_pose_data, str(video_json_path))
                
                # 3. GT 파일 저장 (테스트 세트만)
                if save_gt:
                    start_idx = video_info['label_start_idx']
                    end_idx = video_info['label_end_idx']
                    video_labels = labels[start_idx:end_idx+1]
                    
                    gt_filename = f"{video_name}_gt.npy"
                    gt_path = gt_dir / gt_filename
                    np.save(str(gt_path), video_labels)
                
                video_count += 1
        
        # 전체 메타데이터 저장
        if metadata:
            metadata_path = split_dir / "video_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            saved_files.append("video_metadata.json")
        
        saved_files.append(f"{video_count}개 비디오 JSON 파일")
        if save_gt:
            saved_files.append(f"ground_truth/ ({video_count}개 GT 파일)")
        
        print(f"저장 완료: {prefix}/ - {', '.join(saved_files)}")
    
    def create_full_dataset(self):
        """전체 데이터셋 생성 - 비율 최적화 버전"""
        print("PoseLift 데이터셋 생성 시작! (비율 최적화)")
        
        # 1. 정상 데이터 수집 (카테고리별 모든 데이터)
        normal_videos = self.process_normal_dataset()
        
        # 2. 이상 데이터 수집
        abnormal_videos = self.process_abnormal_dataset()
        
        # 3. 최적 비율 계산 및 데이터 분할
        train_normal_videos, test_normal_videos, train_abnormal_videos, test_abnormal_videos = \
            self.calculate_optimal_ratios(normal_videos, abnormal_videos)
        
        # 4. 훈련 데이터셋 처리 (정상 데이터만 - 이상 탐지)
        print(f"\n훈련 데이터셋 처리 중 (정상 데이터만)...")
        
        # 4-1. 훈련용 정상 데이터만 처리
        print(f"정상 데이터 처리 ({len(train_normal_videos)}개 비디오)")
        train_pose_data, train_labels, train_metadata = self.process_video_batch(
            train_normal_videos, "train_normal", is_abnormal=False
        )
        
        # 이상 데이터는 훈련에 사용하지 않음 (이상 탐지의 기본 원칙)
        if len(train_abnormal_videos) > 0:
            print(f"경고: 훈련에는 정상 데이터만 사용됩니다. 이상 데이터 {len(train_abnormal_videos)}개는 무시됩니다.")
        
        self.save_dataset(train_pose_data, train_labels, "train", save_gt=False, metadata=train_metadata)
        
        # 5. 테스트 데이터셋 처리 (정상 + 이상)
        print(f"\n테스트 데이터셋 처리 중...")
        
        # 5-1. 테스트용 정상 데이터
        print(f"정상 데이터 처리 ({len(test_normal_videos)}개 비디오)")
        test_normal_pose_data, test_normal_labels, test_normal_metadata = self.process_video_batch(
            test_normal_videos, "test_normal", is_abnormal=False
        )
        
        # 5-2. 테스트용 이상 데이터
        print(f"이상 데이터 처리 ({len(test_abnormal_videos)}개 비디오)")
        test_abnormal_pose_data, test_abnormal_labels, test_abnormal_metadata = self.process_video_batch(
            test_abnormal_videos, "test_abnormal", is_abnormal=True
        )
        
        # 5-3. 테스트 데이터 통합
        test_pose_data = {**test_normal_pose_data, **test_abnormal_pose_data}
        test_labels = np.concatenate([test_normal_labels, test_abnormal_labels])
        test_metadata = {**test_normal_metadata, **test_abnormal_metadata}
        
        self.save_dataset(test_pose_data, test_labels, "test", save_gt=True, metadata=test_metadata)
        
        # 6. 데이터셋 요약
        print(f"\n데이터셋 생성 완료!")
        print(f"훈련 데이터: {len(train_pose_data)}명, {len(train_labels)}프레임")
        print(f"  - 정상: {np.sum(train_labels == 0)}프레임 ({np.sum(train_labels == 0)/len(train_labels)*100:.1f}%)")
        print(f"  - 이상: {np.sum(train_labels == 1)}프레임 ({np.sum(train_labels == 1)/len(train_labels)*100:.1f}%)")
        print(f"테스트 데이터: {len(test_pose_data)}명, {len(test_labels)}프레임")
        print(f"  - 정상: {np.sum(test_labels == 0)}프레임 ({np.sum(test_labels == 0)/len(test_labels)*100:.1f}%)")
        print(f"  - 이상: {np.sum(test_labels == 1)}프레임 ({np.sum(test_labels == 1)/len(test_labels)*100:.1f}%)")
        
        # 전체 비율 확인
        total_normal = np.sum(train_labels == 0) + np.sum(test_labels == 0)
        total_abnormal = np.sum(train_labels == 1) + np.sum(test_labels == 1)
        total_frames = total_normal + total_abnormal
        
        print(f"\n전체 데이터셋 비율:")
        print(f"  - 정상:이상 = {total_normal/total_abnormal:.1f}:1 ({total_normal/total_frames*100:.1f}%:{total_abnormal/total_frames*100:.1f}%)")
        print(f"  - Train:Test = {len(train_labels)/len(test_labels):.1f}:1")
        print(f"저장 위치: {self.output_dir.absolute()}")

def main():
    """메인 실행 함수"""
    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed(42)
    np.random.seed(42)
    
    processor = DatasetProcessor()
    
    # 비율 설정 옵션 (필요시 여기서 변경)
    # processor.normal_abnormal_ratio = 0.6  # 6:4 비율로 변경하려면 주석 해제
    
    print(f"설정된 비율:")
    print(f"  - 정상:이상 = {processor.normal_abnormal_ratio:.1%}:{1-processor.normal_abnormal_ratio:.1%}")
    print(f"  - Train:Test = {processor.train_test_ratio}:1")
    
    processor.create_full_dataset()

if __name__ == "__main__":
    main()