import json
import numpy as np
import pickle
from data_preprocess import KeypointPipeline # 곽민준 님의 (수정된) 파이프라인
from tqdm import tqdm
import os

# --- 1. 설정 ---
PIPELINE_CONFIG = {
    "conf_threshold": 0.3,
    "max_gap": 8,
    "smoothing_window_size": 7, # 홀수 권장
    "sequence_window_size": 24, # 논문과 동일하게
    "stride": 4,                # 학습 데이터에만 적용 (평가는 stride=1 추천)
    "norm_method": "bbox",
    "frame_width": 1920,
    "frame_height": 1080,
    "aug_expansion_prob": 0.5, # 학습 데이터에만 적용
    "aug_flip_prob": 0.5,
    "aug_shear_prob": 0.5,
    "aug_shear_range": 0.1
}

# --- 2. 데이터 경로 설정 ---
# (!!!) 이 경로들을 곽민준 님의 실제 경로로 수정하세요 (!!!)
DATA_PATHS = {
    # STG-NF 학습용 (정상 데이터만)
    "normal_train": "path/to/normal_train_jsons/",
    # MLP 학습용 (정상 + 이상)
    "normal_train_mlp": "path/to/normal_train_jsons/", # STG-NF와 동일 데이터 사용
    "abnormal_train_mlp": "path/to/abnormal_train_jsons/",
    # 최종 테스트용 (정상 + 이상)
    "normal_test": "path/to/normal_test_jsons/",
    "abnormal_test": "path/to/abnormal_test_jsons/"
}

# 출력 파일 이름
OUTPUT_PKL_FILES = {
    "normal_train": "normal_train_processed.pkl",
    "normal_train_mlp": "normal_train_mlp_processed.pkl", # MLP용 (증강X, stride=1)
    "abnormal_train_mlp": "abnormal_train_mlp_processed.pkl",
    "normal_test": "normal_test_processed.pkl",
    "abnormal_test": "abnormal_test_processed.pkl"
}

# --- 3. 데이터 로드 함수 (폴더 내 모든 JSON 로드 예시) ---
def load_all_jsons_from_dir(dir_path):
    all_data = {}
    if not os.path.isdir(dir_path):
        print(f"경고: 디렉토리를 찾을 수 없습니다 - {dir_path}")
        return all_data

    print(f"'{dir_path}' 에서 JSON 파일 로드 중...")
    file_list = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    for filename in tqdm(file_list, desc=f"Loading {os.path.basename(dir_path)}"):
        filepath = os.path.join(dir_path, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_data.update(data)
        except Exception as e:
            print(f"파일 로드 오류 ({filename}): {e}")
    print(f"-> 총 {len(all_data)} 명(또는 파일)의 데이터 로드 완료.")
    return all_data

# --- 4. 전처리 실행 ---
global_mean = None
global_std = None

for data_key, json_path in DATA_PATHS.items():
    print(f"\n--- [{data_key}] 데이터 전처리 시작 ---")

    # 4-1. 원본 데이터 로드
    raw_data = load_all_jsons_from_dir(json_path)
    if not raw_data:
        print(f"[{data_key}] 데이터가 비어있어 건너뜁니다.")
        continue

    # 4-2. 파이프라인 설정 조정
    current_config = PIPELINE_CONFIG.copy()
    is_stgnf_train_data = (data_key == "normal_train")

    if not is_stgnf_train_data:
        # STG-NF 학습 외 모든 데이터는 증강 X, stride=1
        current_config["aug_expansion_prob"] = 0.0
        current_config["stride"] = 1
        print("  (증강 비활성화, stride=1 적용)")
    else:
        # STG-NF 학습 데이터는 설정된 stride (e.g., 4) 사용
        current_config["stride"] = PIPELINE_CONFIG["stride"]
        print(f"  (증강 활성화, stride={current_config['stride']} 적용)")


    pipeline = KeypointPipeline(**current_config)

    # 4-3. 전처리 실행
    if is_stgnf_train_data:
        # ★★★ 핵심 ★★★
        # 'normal_train' 데이터로만 평균/표준편차 계산 (fit)
        sequences = pipeline.process_for_training(raw_data)
        global_mean = pipeline.mean_coord
        global_std = pipeline.std_coord
        print(f"  -> 글로벌 통계치 계산 완료: Mean={global_mean:.4f}, Std={global_std:.4f}")
    else:
        # 다른 데이터들은 'normal_train'의 통계치 주입 후 처리 (transform)
        if global_mean is None or global_std is None:
            print("오류: 'normal_train'을 먼저 처리하여 평균/표준편차를 계산해야 합니다.")
            exit()
        pipeline.mean_coord = global_mean
        pipeline.std_coord = global_std

        # process_for_evaluation 사용 (증강X, fit_standardizer() 생략)
        sequences = pipeline.process_for_evaluation(raw_data)

    # 4-4. 결과 저장
    output_path = OUTPUT_PKL_FILES[data_key]
    print(f"  -> [{data_key}] 전처리된 시퀀스 {len(sequences)}개 저장 중... ({output_path})")

    # 저장할 데이터 구조 (평균/표준편차도 함께 저장)
    data_to_save = {
        "sequences": sequences,
        "global_mean": global_mean,
        "global_std": global_std,
        "num_joints_total": 18
    }
    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f)

print("\n--- 모든 데이터 전처리 완료 ---")