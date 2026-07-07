# -*- coding: utf-8 -*-
# @Description: Main process of network testing.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import os, time, sys, gc, cv2, logging, errno  # NOQA
import numpy as np
import torch
import torch.nn as nn  # NOQA
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets.data_io import *
from datasets.dtu import DTUDataset
from datasets.tnt import TNTDataset

from models.mvsnet import MVSNet
from models.utils import *
from models.utils.opts import get_opts
# from model_analysis import measure_memory_and_fps


os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
cudnn.benchmark = True

args = get_opts()


def test():
    total_time = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()
            outputs = model(sample_cuda, "test")

            end_time = time.time()
            total_time += end_time - start_time
            outputs = tensor2numpy(outputs)
            del sample_cuda

            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(args.levels)].numpy()
            imgs = sample["imgs"]
            logger.info('Iter {}/{}, Time:{:.3f} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

            for filename, cam, img, depth_est, depth3, depth2, depth1, photometric_confidence, pc3, pc2, pc1 in zip(filenames, cams, imgs,
                                                                                                                    outputs["depth"],
                                                                                                                    outputs["stage3"]["depth"],
                                                                                                                    outputs["stage2"]["depth"],
                                                                                                                    outputs["stage1"]["depth"],
                                                                                                                    outputs["photometric_confidence"],
                                                                                                                    outputs["stage3"]["photometric_confidence"],
                                                                                                                    outputs["stage2"]["photometric_confidence"],
                                                                                                                    outputs["stage1"]["photometric_confidence"]):

                depth_filename3 = os.path.join(args.outdir, filename.format('depth_est', '_stage3.pfm'))
                depth_filename2 = os.path.join(args.outdir, filename.format('depth_est', '_stage2.pfm'))
                depth_filename1 = os.path.join(args.outdir, filename.format('depth_est', '_stage1.pfm'))

                h, w = photometric_confidence.shape
                pc3 = cv2.resize(pc3, (w, h), interpolation=cv2.INTER_NEAREST)
                pc2 = cv2.resize(pc2, (w, h), interpolation=cv2.INTER_NEAREST)
                pc1 = cv2.resize(pc1, (w, h), interpolation=cv2.INTER_NEAREST)
                confidence_filename3 = os.path.join(args.outdir, filename.format('confidence', '_stage3.pfm'))
                confidence_filename2 = os.path.join(args.outdir, filename.format('confidence', '_stage2.pfm'))
                confidence_filename1 = os.path.join(args.outdir, filename.format('confidence', '_stage1.pfm'))
                img = img[0].numpy()  # ref view
                cam = cam[0]  # ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                # ply_filename = os.path.join(self.args.outdir, filename.format('ply_local', '.ply'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                # os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                save_pfm(depth_filename3, depth3)
                save_pfm(depth_filename2, depth2)
                save_pfm(depth_filename1, depth1)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                save_pfm(confidence_filename3, pc3)
                save_pfm(confidence_filename2, pc2)
                save_pfm(confidence_filename1, pc1)
                if args.display:
                    depth_color = visualize_depth(depth_est)
                    cv2.imwrite(os.path.join(args.outdir, filename.format('depth_est', '.jgp')), depth_color)
                    # save confidence maps
                    cv2.imwrite(os.path.join(args.outdir, filename.format('confidence', '.jgp')), visualize_depth(photometric_confidence))
                    cv2.imwrite(os.path.join(args.outdir, filename.format('confidence', '_1.jgp')), visualize_depth(pc1))
                    cv2.imwrite(os.path.join(args.outdir, filename.format('confidence', '_2.jgp')), visualize_depth(pc2))
                # inter_val = outputs["stage4"]["interval"]
            # for filename, cam, img, depth_est, photometric_confidence in zip(filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"]):
            #     img = img[0].numpy()    # ref view
            #     cam = cam[0]            # ref cam

            #     depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
            #     confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
            #     cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
            #     img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
            #     os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
            #     os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
            #     if args.which_dataset == 'dtu':
            #         os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
            #         os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)

            #     # save confidence and depth maps
            #     depth_list = [outputs['stage{}'.format(i)]['depth'].squeeze(0) for i in range(1, 5)]
            #     confidence_list = [outputs['stage{}'.format(i)]['photometric_confidence'].squeeze(0) for i in range(1, 5)]
            #     depth_est = depth_list[-1]
            #     photometric_confidence = confidence_list[-1]
            #     if not args.save_all_stages:
            #         save_pfm(depth_filename, depth_est)
            #         save_pfm(confidence_filename, photometric_confidence)
            #     else:
            #         for stage_idx, depth_est in enumerate(depth_list):
            #             if stage_idx != args.levels - 1:
            #                 depth_filename = os.path.join(args.outdir, filename.format('depth_est', "_stage" + str(stage_idx + 1) + '.pfm'))
            #             else:
            #                 depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
            #             save_pfm(depth_filename, depth_est)
            #         for stage_idx, photometric_confidence in enumerate(confidence_list):
            #             if stage_idx != args.levels - 1:
            #                 confidence_filename = os.path.join(args.outdir, filename.format('confidence', "_stage" + str(stage_idx + 1) + '.pfm'))
            #             else:
            #                 confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
            #             save_pfm(confidence_filename, photometric_confidence)

                # save cams, img
                # if args.which_dataset == 'dtu':
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

    torch.cuda.empty_cache()
    gc.collect()
    return total_time, len(TestImgLoader)


def initLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))

    if args.which_dataset == 'tnt':
        logfile = os.path.join(args.logdir, 'TNT-test-' + curTime + '.log')
    else:
        logfile = os.path.join(args.logdir, 'test-' + curTime + '.log')
    # add by liyi, used for creat logfile
    if not os.path.exists(os.path.dirname(logfile)):
        try:
            os.makedirs(os.path.dirname(logfile))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    if not args.nolog:
        fileHandler = logging.FileHandler(logfile, mode='a')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.info("Logger initialized.")
    logger.info("Writing logs to file: {}".format(logfile))
    logger.info("Current time: {}".format(curTime))

    settings_str = "All settings:\n"
    for k, v in vars(args).items():
        settings_str += '{0}: {1}\n'.format(k, v)
    logger.info(settings_str)

    return logger


if __name__ == '__main__':
    logger = initLogger()

    # dataset, dataloader
    if args.which_dataset == 'dtu':
        test_dataset = DTUDataset(args.testpath, args.testlist, "test", args.n_views, max_wh=(1600, 1200))  # ,1152, 864
    elif args.which_dataset == 'tnt':
        test_dataset = TNTDataset(args.testpath, args.testlist, split=args.split, n_views=args.n_views, img_wh=(-1, 1024), cam_mode=args.cam_mode, img_mode=args.img_mode)

    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # @Note MVSNet model
    model = MVSNet(args)

    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict['model'])

    model.cuda()
    model.eval()

    test()
