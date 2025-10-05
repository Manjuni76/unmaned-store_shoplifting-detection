import json
import os
import torch

# ★ 팔 관절 인덱스 (COCO18 기준: [R-Shoulder, R-Elbow, R-Wrist, L-Shoulder, L-Elbow, L-Wrist])
#   dataset.py에 COCO18_ARMS가 있으면 import, 없으면 fallback 사용
try:
    from dataset import COCO18_ARMS
except Exception:
    COCO18_ARMS = [2, 3, 4, 5, 6, 7]

def init_model_params(args, dataset):
    # 안전 검사: dataset이 dict인지 확인
    if not isinstance(dataset, dict):
        print(f"[UTILS-ERROR] init_model_params에 예상치 못한 타입 전달: {type(dataset)}. 함수 호출 시 전달된 값을 확인하세요.")
        # 시그니처 호환을 위해 기본값 사용
        # (C, T, V) 기본값: 채널=3, T=args.seg_len 또는 24, V=18
        C = 3
        T = getattr(args, 'seg_len', 24)
        V = 18
        pose_shape = (C, T, V)
    else:
        # (C, T, V): 관절 부위에 맞게 사용
        try:
            pose_shape = dataset["test"][0][0].shape if args.model_confidence else dataset["test"][0][0][:2].shape
        except Exception as e:
            print(f"[UTILS-WARN] dataset에서 pose_shape를 읽는 중 오류: {e}. 기본값 사용")
            C = 3
            T = getattr(args, 'seg_len', 24)
            V = 18
            pose_shape = (C, T, V)
    
    # subset_idx는 train_eval.py에서 직접 설정됨 (우선순위 높음)
    subset_idx = None
    
    # 디버그용 출력
    print(f"[UTILS] pose_shape: {pose_shape}, joint_subset: {getattr(args, 'joint_subset', None)}")

    return {
        'pose_shape': pose_shape,
        'hidden_channels': args.model_hidden_dim,
        'K': args.K,
        'L': args.L,
        'R': args.R,
        'actnorm_scale': 1.0,
        'flow_permutation': args.flow_permutation,
        'flow_coupling': 'affine',
        'LU_decomposed': True,
        'learn_top': False,
        'edge_importance': args.edge_importance,
        'temporal_kernel_size': args.temporal_kernel,
        'strategy': args.adj_strategy,
        'max_hops': args.max_hops,
        'device': args.device,
        'subset_idx': subset_idx,  # 모델에 전달
    }

def dump_args(args, ckpt_dir):
    path = os.path.join(ckpt_dir, "args.json")
    data = vars(args)
    with open(path, 'w') as fp:
        json.dump(data, fp)

def calc_reg_loss(model, reg_type='l2', avg=True):
    reg_loss = None
    parameters = list(param for name, param in model.named_parameters() if 'bias' not in name)
    num_params = len(parameters)
    if reg_type.lower() == 'l2':
        for param in parameters:
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param ** 2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2) ** 2
        if avg:
            reg_loss /= num_params
        return reg_loss
    else:
        return torch.tensor(0.0, device=model.device)

def get_fn_suffix(args):
    fn_suffix = args.dataset + args.conv_oper
    return fn_suffix

def csv_log_dump(args, log_dict):
    try:
        date_time = args.ckpt_dir.split('/')[-3]  # 'Aug20_0839'
        date_str, time_str = date_time.split('_')[:2]
    except:
        date_time = 'parse_fail'
        date_str, time_str = '??', '??'
    param_arr = [date_str, time_str, args.seed, args.conv_oper, args.num_transform, args.norm_scale,
                 args.prop_norm_scale, args.seg_stride, args.seg_len, args.patch_features, args.patch_size,
                 args.ae_optimizer, args.ae_sched, args.dropout, args.ae_batch_size,
                 args.ae_epochs, args.ae_lr, args.ae_lr_decay, args.ae_weight_decay, log_dict['F_ae_loss'],
                 args.n_clusters, args.dcec_batch_size, args.dcec_epochs, args.dcec_optimizer, args.dcec_sched,
                 args.dcec_lr, args.dcec_lr_decay, args.alpha, args.gamma, args.update_interval,
                 log_dict['F_delta_labels'], log_dict['F_dcec_loss'], log_dict['dp_auc'], log_dict['F_ae_num_params'],
                 log_dict['F_dcec_num_params'], args.headless]

    res_str = '_{}'.format(int(10 * log_dict['dp_auc']))
    log_template = len(param_arr) * '{}, ' + '\n'
    log_str = log_template.format(*param_arr)
    debug_str = '_debug' if args.debug else ''
    csv_path = os.path.join(args.ckpt_dir, '{}{}{}_log_dump.csv'.format(date_time, debug_str, res_str))
    with open(csv_path, 'w') as csv_file:
        csv_file.write(log_str)

def calc_num_of_params(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("Number of params in net: {}K".format(num_params / 1e3))
    return num_params / 1e3
