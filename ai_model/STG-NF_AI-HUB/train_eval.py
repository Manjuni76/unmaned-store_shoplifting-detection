import random
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params
#
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
import subprocess
import sys

# 관절 부위 매핑은 dataset.py의 JOINT_SUBSET_MAP을 재사용하여 일관성 유지
from dataset import JOINT_SUBSET_MAP as JOINT_SUBSETS

def set_seed(seed=42):
    """모든 랜덤 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 재현성 우선
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[SEED] 모든 랜덤 시드를 {seed}로 고정했습니다.")

def main():
    #모델에 사용하는 인자값들 설정 terminal에서 돌릴 때 설정도 가능함
    parser = init_parser()
    args = parser.parse_args()
    
    # 사용자에게 관절 부위를 입력받음
    print("\n사용할 관절 부위를 선택하세요:")
    print("- arms: 전체 팔 [2,3,4,5,6,7]")
    print("- legs: 전체 다리 [8,9,10,11,12,13]")
    print("- left_arm: 왼쪽팔만 [5,6,7]")
    print("- right_arm: 오른쪽팔만 [2,3,4]")
    print("- left_leg: 왼쪽다리만 [11,12,13]")
    print("- right_leg: 오른쪽다리만 [8,9,10]")
    print("- body: 몸통 관절 [1,2,5,8,11]")
    print("- head: 머리 관절 [0,14,15,16,17]")
    print("- upper: 상체 전체")
    print("- lower: 하체 전체")
    print("- all: 모든 관절 [0-17]")
    
    # 사용할 관절 선택
    while True:
        subset_choice = input("\n선택 (arms/legs/left_arm/right_arm/left_leg/right_leg/body/head/upper/lower/all): ").strip().lower() # 입력 앞뒤 공백 제거, 대문자가 들어와도 소문자로 바꿔주게 함
        if subset_choice in JOINT_SUBSETS:
            break
        print(f"잘못된 선택입니다. {list(JOINT_SUBSETS.keys())} 중 하나를 입력해주세요.")
    
    # 선택한 관절 부위 적용
    args.joint_subset = subset_choice
    subset_idx = JOINT_SUBSETS[subset_choice]  # 관절 인덱스 배열 가져오기
    print(f"\n[INFO] 선택한 관절 부위: {subset_choice}")
    print(f"[INFO] 사용할 관절 인덱스: {subset_idx if subset_idx else '전체(모든 관절)'}")
    
    # 시드 설정 - 사용자 입력 받기
    while True:
        try:
            seed_input = input("\n사용할 시드 값을 입력하세요 (양의 정수, 기본값 42): ").strip()
            if not seed_input:  # 입력이 없으면 기본값 사용
                FIXED_SEED = 42
                break
            FIXED_SEED = int(seed_input)
            if FIXED_SEED > 0:
                break
            print("양의 정수를 입력해주세요.")
        except ValueError:
            print("올바른 정수를 입력해주세요.")
    
    print(f"[SEED] 랜덤 시드를 {FIXED_SEED}로 설정합니다.")
    set_seed(FIXED_SEED)
    
    # ========== 기존 시드 설정 코드 (주석처리) ==========
    # if args.seed == 999:  # Record and init seed
    #     args.seed = torch.initial_seed()
    #     np.random.seed(0)
    # else:
    #     random.seed(args.seed)
    #     torch.backends.cudnn.deterministic = True #연산 결과 항상 동일하게
    #     torch.backends.cudnn.benchmark = True #GPU 성능 최적화
    #     torch.manual_seed(args.seed)
    #     np.random.seed(0)
    # ========== 기존 시드 설정 코드 끝 ==========

    args, model_args = init_sub_args(args) # 서브 인자 초기화
    # 시드와 관절 부위 정보를 디렉토리 이름에 포함하여 저장
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset, 
                                   joint_subset=subset_choice, 
                                   seed=FIXED_SEED) # 체크포인트, 로그 저장 위치 생성

    pretrained = vars(args).get('checkpoint', None) # 기존 모델 불러오기 없으면 새로 학습 시작

    dataset, loader = get_dataset_and_loader(args, trans_list, only_test=args.only_test) # 데이터셋 불러오기

    # ★ 확인 프린트
    if 'train' in dataset and dataset['train'] is not None:
        tr = dataset['train']
        print(f"[CHECK] train: segs_data_np shape=(N={tr.segs_data_np.shape[0]}, C={tr.segs_data_np.shape[1]}, T={tr.segs_data_np.shape[2]}, V={tr.segs_data_np.shape[3]}), segs_score_np shape={tr.segs_score_np.shape}")
    if 'test' in dataset and dataset['test'] is not None:
        te = dataset['test']
        print(f"[CHECK] test:  segs_data_np shape=(N={te.segs_data_np.shape[0]}, C={te.segs_data_np.shape[1]}, T={te.segs_data_np.shape[2]}, V={te.segs_data_np.shape[3]}), segs_score_np shape={te.segs_score_np.shape}")

    print(f"[CHECK] joint_subset={getattr(args, 'joint_subset', None)}, headless={getattr(args, 'headless', None)}")
    print(f"[UTILS] device: {args.device}")

    model_args = init_model_params(args, dataset) #모델 인자값 받아옴
    
    # 선택한 관절 인덱스를 모델 인자에 추가
    if subset_choice != "all":
        model_args['subset_idx'] = subset_idx

    model = STG_NF(**model_args) # 모델 초기화
    num_of_params = calc_num_of_params(model)
    print(f"[MODEL-INFO] 모델 파라미터 수: {num_of_params:,}, 선택된 관절 부위: {subset_choice}, 관절 수: {len(subset_idx) if subset_idx else 18}")
    #모델 설정 (Optimizer, schedduler 설정)
    trainer = Trainer(args, model, loader['train'], loader['test'], 
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    # 사전 학습된 모델이 있으면 불러오기
    if pretrained:
        trainer.load_checkpoint(pretrained)
    # 없으면 새로 학습 시작
    else:
        writer = SummaryWriter()
        trainer.train(log_writer=writer) #train 시작
        dump_args(args, args.ckpt_dir) # arg 설정값 저장

    #Testing and scoring:
    normality_scores = trainer.test() # test 수행

    # evaluation 수행
    auc_roc, scores_np, auc_pr, eer, eer_threshold = score_dataset(normality_scores, dataset["test"].metadata, args=args) 
 
    # Logging and recording results
    print("\n-------------------------------------------------------")
    print(f'결과 ({subset_choice} 부위):')
    print('auc(roc): {}'.format(auc_roc))
    print('auc(pr): {}'.format(auc_pr))
    print('eer: {}'.format(eer))
    print('eer threshold: {}'.format(eer_threshold))
    print('Number of samples', scores_np.shape[0])
   


if __name__ == '__main__':
    main()
