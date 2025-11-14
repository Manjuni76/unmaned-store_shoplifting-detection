"""
GT 파일 존재 여부 확인 스크립트 (코드 실행 없이 정보만 제공)
"""

import json
import os

def check_gt_files():
    base_dir = r"C:\Users\User\Desktop\Sejong\Under_Graduate\3_2\파이썬기반딥러닝\unmaned_store_shoplifting_detection"
    
    # JSON 파일 경로
    mlp_train_json = os.path.join(base_dir, "data_split", "output", "mlp_train_data.json")
    test_json = os.path.join(base_dir, "data_split", "output", "test_data.json")
    
    # GT 디렉토리 경로
    mlp_gt_dir = os.path.join(base_dir, "data", "gt", "mlp_train_gt")
    test_gt_dir = os.path.join(base_dir, "data", "gt", "test_gt")
    
    print("="*70)
    print("GT 파일 존재 여부 확인")
    print("="*70)
    
    # 1. MLP Train abnormal 파일 확인
    print("\n[1] MLP Train Abnormal 데이터 GT 확인")
    print("-" * 70)
    with open(mlp_train_json, 'r', encoding='utf-8') as f:
        mlp_data = json.load(f)
    
    abnormal_videos = mlp_data.get('abnormal', [])
    print(f"Total abnormal videos: {len(abnormal_videos)}")
    
    # GT 파일 리스트
    mlp_gt_files = set(os.listdir(mlp_gt_dir))
    
    missing_mlp = []
    existing_mlp = []
    
    for video in abnormal_videos:
        filename = video['filename'].replace('.mp4', '.npy')
        if filename in mlp_gt_files:
            existing_mlp.append(filename)
        else:
            missing_mlp.append(filename)
    
    print(f"✅ GT에 있는 파일: {len(existing_mlp)}/{len(abnormal_videos)}")
    print(f"❌ GT에 없는 파일: {len(missing_mlp)}/{len(abnormal_videos)}")
    
    if missing_mlp:
        print("\n누락된 GT 파일 (처음 10개):")
        for i, fname in enumerate(missing_mlp[:10], 1):
            print(f"  {i}. {fname}")
        if len(missing_mlp) > 10:
            print(f"  ... 외 {len(missing_mlp) - 10}개")
    
    # 2. Test abnormal 파일 확인
    print("\n" + "="*70)
    print("[2] Test Abnormal 데이터 GT 확인")
    print("-" * 70)
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    abnormal_videos = test_data.get('abnormal', [])
    print(f"Total abnormal videos: {len(abnormal_videos)}")
    
    # GT 파일 리스트
    test_gt_files = set(os.listdir(test_gt_dir))
    
    missing_test = []
    existing_test = []
    
    for video in abnormal_videos:
        filename = video['filename'].replace('.mp4', '.npy')
        if filename in test_gt_files:
            existing_test.append(filename)
        else:
            missing_test.append(filename)
    
    print(f"✅ GT에 있는 파일: {len(existing_test)}/{len(abnormal_videos)}")
    print(f"❌ GT에 없는 파일: {len(missing_test)}/{len(abnormal_videos)}")
    
    if missing_test:
        print("\n누락된 GT 파일 (처음 10개):")
        for i, fname in enumerate(missing_test[:10], 1):
            print(f"  {i}. {fname}")
        if len(missing_test) > 10:
            print(f"  ... 외 {len(missing_test) - 10}개")
    
    # 3. 요약
    print("\n" + "="*70)
    print("[요약]")
    print("="*70)
    print(f"MLP Train:")
    print(f"  - Total abnormal: {len(mlp_data.get('abnormal', []))}")
    print(f"  - GT 존재: {len(existing_mlp)}")
    print(f"  - GT 누락: {len(missing_mlp)}")
    print(f"  - 커버리지: {len(existing_mlp)/len(mlp_data.get('abnormal', []))*100:.1f}%")
    
    print(f"\nTest:")
    print(f"  - Total abnormal: {len(test_data.get('abnormal', []))}")
    print(f"  - GT 존재: {len(existing_test)}")
    print(f"  - GT 누락: {len(missing_test)}")
    print(f"  - 커버리지: {len(existing_test)/len(test_data.get('abnormal', []))*100:.1f}%")
    
    # 4. 예시 파일명 매칭 확인
    print("\n" + "="*70)
    print("[파일명 매칭 예시]")
    print("="*70)
    if len(existing_mlp) > 0:
        example = existing_mlp[0]
        print(f"✅ 매칭 성공 예시:")
        print(f"   Video: {example.replace('.npy', '.mp4')}")
        print(f"   GT:    {example}")
    
    if len(missing_mlp) > 0:
        example = missing_mlp[0]
        print(f"\n❌ 매칭 실패 예시:")
        print(f"   Video: {example.replace('.npy', '.mp4')}")
        print(f"   GT:    {example} (NOT FOUND)")

if __name__ == "__main__":
    check_gt_files()
