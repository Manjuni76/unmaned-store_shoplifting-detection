#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
여러 JSON 파일에서 비디오 경로를 읽어 skeleton pose 데이터를 추출하여 저장하는 스크립트
"""
import os
import json
from pathlib import Path
from yolo_skeleton_extractor import YOLOSkeletonExtractor
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 처리할 JSON 파일 목록
JSON_FILES = [
    'C:\\Users\\User\\Desktop\\Sejong\\Under_Graduate\\3_2\\파이썬기반딥러닝\\unmaned_store_shoplifting_detection\\data_split\\output\\test_data.json',
    'C:\\Users\\User\\Desktop\\Sejong\\Under_Graduate\\3_2\\파이썬기반딥러닝\\unmaned_store_shoplifting_detection\\data_split\\output\\mlp_train_data.json',
    'C:\\Users\\User\\Desktop\\Sejong\\Under_Graduate\\3_2\\파이썬기반딥러닝\\unmaned_store_shoplifting_detection\\data_split\\output\\train_data.json',
]

# 작업 폴더 (skeleton 데이터 저장)
BASE_OUT_DIR = Path('C:\\Users\\User\\Desktop\\Sejong\\Under_Graduate\\3_2\\파이썬기반딥러닝\\unmaned_store_shoplifting_detection\\data')


def process_video_task(args):
    video_path, out_json = args
    try:
        extractor = YOLOSkeletonExtractor(conf_threshold=0.3, kpt_threshold=0.1)
        skeleton_data = extractor.process_video(video_path, str(out_json))
        return (video_path, True, None)
    except Exception as e:
        return (video_path, False, str(e))



if __name__ == "__main__":
    for json_file in JSON_FILES:
        json_path = Path(json_file)
        if not json_path.exists():
            print(f"파일 없음: {json_file}")
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        video_list = data.get('normal', [])
        out_dir = BASE_OUT_DIR / f"{json_path.stem}_skeleton_data"
        out_dir.mkdir(exist_ok=True)
        print(f"{json_file} → {out_dir} 저장 시작, 총 {len(video_list)}개")

        tasks = []
        for video_info in video_list:
            video_path = video_info['full_path']
            filename = video_info['filename']
            video_name = os.path.splitext(filename)[0]
            out_json = out_dir / f"{video_name}_skeleton.json"
            tasks.append((video_path, out_json))

        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_video_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="진행률"):
                video_path, success, error = future.result()
                fname = os.path.basename(video_path)
                if success:
                    print(f"완료: {fname}")
                else:
                    print(f"실패: {fname} → {error}")
    print("모든 작업 완료.")
