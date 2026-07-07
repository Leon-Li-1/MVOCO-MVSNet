# -*- coding: utf-8 -*-


import argparse


def get_opts():
    parser = argparse.ArgumentParser(description="args")

    # global settings0000
    parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'val'])
    parser.add_argument('--which_dataset', default='dtu', choices=['dtu', 'tnt', 'blendedmvs'], help='which dataset for using')
    parser.add_argument('--which_module', default='mvoco_plus', help='mvoco or mvoco_plus')
    parser.add_argument('--n_views', type=int, default=5, help='num of view')  # dtu 5; tnt: 11; Bleendmvs:7 试一试9

    parser.add_argument('--levels', type=int, default=4, help='num of stages')
    parser.add_argument('--hypo_plane_num_stages', type=str, default="8,8,4,4", help='num of hypothesis planes for each stage')  # train 8,8,4,4 dtu_test8,8,8,4  tnt_test 16,8,8,4
    parser.add_argument('--depth_interal_ratio_stages', type=str, default="0.5,0.5,0.5,0.5", help='depth interals for each stage')  # bld "0.5,0.5,0.5,1"
    parser.add_argument("--feat_base_channel", type=int, default=8, help='channel num for base feature')
    parser.add_argument("--reg_base_channel", type=int, default=8, help='channel num for regularization')
    parser.add_argument('--group_cor_dim_stages', type=str, default="8,8,8,8", help='group correlation dim')
    parser.add_argument('--cs_fband', type=float, default=8.0, )  # sine-cosine moudle 频率带宽
    parser.add_argument('--real_depth', type=bool, default=False, help='robust training')

    parser.add_argument('--data_scale', default='raw', type=str, choices=['mid', 'raw'], help='use mid or raw resolution')  # test dtu use raw mode, train use mid
    parser.add_argument('--trainpath', default="/home/lida/lida/2Ddataset/dtu_data/DTU/dtu_training", help='data path for training')
    parser.add_argument('--testpath', default="/home/lida/lida/2Ddataset/dtu_data/dtu-test-1200", help='data path for testing')  # "/home/lida/lida/2Ddataset/dtu_data/dtu-test-1200"
    parser.add_argument('--trainlist', default="./datasets/lists/dtu/train.txt", help='data list for training')
    parser.add_argument('--testlist', default="./datasets/lists/dtu/test.txt", help='data list for testing')  # test_copy
    # parser.add_argument('--trainpath', default="/home/lida/lida/2Ddataset/blendedmvs/dataset_low_res", help='data path for training')
    # parser.add_argument('--testpath', default="/home/lida/lida/2Ddataset/tankandtemples", help='data path for testing')  # "/home/lida/lida/2Ddataset/tankandtemples"
    # parser.add_argument('--trainlist', default="./datasets/lists/blendedmvs/low_res_all.txt", help='data list for training')
    # parser.add_argument('--testlist', default="./datasets/lists/tnt/advanced.txt", help='data list for testing')  # "./datasets/lists/blendedmvs/val.txt"  "./datasets/lists/tnt/intermediate.txt"

    # training configd
    parser.add_argument('--stage_lw', type=str, default=[1.5, 1.5, 1.5, 1.5], help='loss weight for different stages')  # "1,1,1,1"
    parser.add_argument('--refine_stage_lw', type=str, default=[1., 1., 1.], help='refined depth loss weight for different stages')  # "1,1,1,1"
    parser.add_argument('--wsup', type=float, default=1)  # 8.0 0.8 0.001已经测试过
    parser.add_argument('--wrefine', type=float, default=0)
    parser.add_argument('--wrecon', type=float, default=8.0)  # 8.0 0.8
    parser.add_argument('--wssim', type=float, default=6.0)   # 6.0
    parser.add_argument('--wsmooth', type=float, default=0.18)  # 去掉深度的平滑损失 0.18

    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
    parser.add_argument('--lr_scheduler', type=str, default='MS', help='scheduler for learning rate')  # onecycle  MS
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # bld:0.0005 dtu:0.001
    parser.add_argument('--lrepochs', type=str, default="6,10,12,14:2", help='epoch ids to downscale lr and the downscale rate')  # "1,3,5,7,9,11,13,15:1.5"
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--summary_freq', type=int, default=100, help='print and summary frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
    parser.add_argument('--eval_freq', type=int, default=1, help='eval frequency')

    # depth ground-truth mask_rate, whitch will mask valid pixel of depth ground-truth 设置把深度值置为无效的概率,以测试MVOCO的鲁棒性
    parser.add_argument('--mask_rate', type=float, default=0.0, help='depth ground-truth mask_rate based on ground-truth mask')

    parser.add_argument('--robust_train', type=bool, default=True, help='robust training')

    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')  # "./outputs/dtu/jiayou_train/model_15.ckpt"
    parser.add_argument('--resume', default=False, action='store_true', help='continue to train the model')

# log config
    parser.add_argument('--outdir', default='./outputs/dtu/jiayou_train_allnorf', help='output dir')  # jiayou_test  jiayou_train
    parser.add_argument('--logdir', default='./outputs/dtu/jiayou_train_allnorf', help='the directory to save checkpoints/logs')
    parser.add_argument('--nolog', action='store_true', help='do not log into .log file')
    parser.add_argument('--notensorboard', action='store_true', help='do not log into tensorboard')

# testing config
    parser.add_argument('--loadckpt', default='./outputs/dtu/jiayou_train_seed123/model_15.ckpt', help='load a specific checkpoint')
    parser.add_argument('--split', type=str, default='advanced', choices=['intermediate', 'advanced'], help='intermediate|advanced for tanksandtemples')
    parser.add_argument('--img_mode', type=str, default='resize', choices=['resize', 'crop'], help='image resolution matching strategy for TNT dataset')
    parser.add_argument('--cam_mode', type=str, default='origin', choices=['origin', 'short_range'], help='camera parameter strategy for TNT dataset')
    parser.add_argument('--save_all_stages', default=True, action='store_true', help='save confidence maps for all stages')
    parser.add_argument('--display', default=False, action='store_true', help='continue to train the model')

    # pytorch config
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
    parser.add_argument('--pin_m', default=False, action='store_true', help='data loader pin memory')
    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()
