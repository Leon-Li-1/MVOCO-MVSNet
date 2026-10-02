# -*- coding: utf-8 -*-
# @Description: Point cloud fusion strategy for DTU dataset based on Open3D Library.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import torch
import numpy as np
import sys
import argparse
import errno, os  # NOQA
import glob  # NOQA
import os.path as osp  # NOQA
import re
import cv2
from PIL import Image
import gc
import open3d as o3d

import torch  # NOQA
import torch.nn.functional as F
import numpy as np  # NOQA


parser = argparse.ArgumentParser(description='Depth fusion with consistency check.')
parser.add_argument('--root_path', type=str, default='/home/lida/lida/2Ddataset/dtu_data/dtu-test-1200')
parser.add_argument('--depth_path', type=str, default='./outputs/dtu/jiayou_test')
parser.add_argument('--data_list', type=str, default='datasets/lists/dtu/test.txt')  # test_copy
parser.add_argument('--ply_path', type=str, default='./outputs/dtu/jiayou_test/open3d_fusion_plys')
# parser.add_argument('--dist_thresh', type=float, default=0.1)
# parser.add_argument('--prob_thresh', type=float, default=0.3)
# parser.add_argument('--num_consist', type=int, default=3)
parser.add_argument('--device', type=str, default='cuda')  # cpu   cuda

args = parser.parse_args()


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values.view(-1, 1, 1, height * width)  # [B, 3, 1, H*W]

        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, height, width)

    return warped_src_fea


def generate_points_from_depth(depth, proj):
    '''
    :param depth: (B, 1, H, W)
    :param proj: (B, 4, 4)
    :return: point_cloud (B, 3, H, W)
    '''
    batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
    inv_proj = torch.inverse(proj)

    rot = inv_proj[:, :3, :3]  # [B,3,3]
    trans = inv_proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz * depth.view(batch, 1, -1)
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, H*W]
    proj_xyz = proj_xyz.view(batch, 3, height, width)

    return proj_xyz


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()


def write_ply(file, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    o3d.io.write_point_cloud(file, pcd, write_ascii=False)


def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    '''
    :param ref_depth: (1, 1, H, W)
    :param src_depths: (B, 1, H, W)
    :param ref_proj: (1, 4, 4)
    :param src_proj: (B, 4, 4)
    :return: ref_pc: (1, 3, H, W), aligned_pcs: (B, 3, H, W), dist: (B, 1, H, W)
    '''

    ref_pc = generate_points_from_depth(ref_depth, ref_proj)
    src_pcs = generate_points_from_depth(src_depths, src_projs)

    aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0])**2
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1])**2
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2])**2
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

    return ref_pc, aligned_pcs, dist


def parse_cameras(path):
    cam_txt = open(path).readlines()
    f = lambda xs: list(map(lambda x: list(map(float, x.strip().split())), xs))

    extr_mat = f(cam_txt[1:5])
    intr_mat = f(cam_txt[7:10])

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    # ---- 新增：读取 depth 范围 ----
    depth_line = cam_txt[11].strip().split()
    depth_max = float(depth_line[0])
    depth_min = float(depth_line[1])

    # DTU dataset 特殊处理
    if depth_max > 425:
        depth_max = 935
        depth_min = 425
    # --------------------------------

    return extr_mat, intr_mat, depth_max, depth_min


def load_data(root_path, depth_path, scene_name, thresh):

    depths = []
    projs = []
    rgbs = []

    for view in range(49):
        img_filename = "{}/{}/images/{:08d}.jpg".format(depth_path, scene_name, view)
        cam_filename = "{}/{}/cams/{:08d}_cam.txt".format(depth_path, scene_name, view)
        depth_filename = "{}/{}/depth_est/{:08d}.pfm".format(depth_path, scene_name, view)
        confidence_filename = "{}/{}/confidence/{:08d}.pfm".format(depth_path, scene_name, view)
    
        extr_mat, intr_mat, depth_max, depth_min = parse_cameras(cam_filename)
        proj_mat = np.eye(4)
        proj_mat[:3, :4] = np.dot(intr_mat[:3, :3], extr_mat[:3, :4])
        projs.append(torch.from_numpy(proj_mat))
    
        dep_map, _ = read_pfm(depth_filename)
        h, w = dep_map.shape
        conf_map, _ = read_pfm(confidence_filename)
        conf_map = cv2.resize(conf_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
        # ✅ 组合过滤：置信度 + 深度范围
        valid_mask = (conf_map > thresh) & (dep_map >= depth_min) & (dep_map <= depth_max)
        dep_map = dep_map * valid_mask.astype(np.float32)
    
        depths.append(torch.from_numpy(dep_map).unsqueeze(0))
    
        rgb = np.array(Image.open(img_filename))
        rgbs.append(rgb)

    depths = torch.stack(depths).float()
    projs = torch.stack(projs).float()
    if args.device == 'cuda' and torch.cuda.is_available():
        depths = depths.cuda()
        projs = projs.cuda()

    return depths, projs, rgbs


def extract_points(pc, mask, rgb):
    pc = pc.cpu().numpy()
    mask = mask.cpu().numpy()

    mask = np.reshape(mask, (-1,))
    pc = np.reshape(pc, (-1, 3))
    rgb = np.reshape(rgb, (-1, 3))

    points = pc[np.where(mask)]
    colors = rgb[np.where(mask)]

    points_with_color = np.concatenate([points, colors], axis=1)

    return points_with_color


def open3d_filter(dist_thresh, prob_thresh, num_consist):
    with torch.no_grad():
        mkdir_p(args.ply_path)
        all_scenes = open(args.data_list, 'r').readlines()
        all_scenes = list(map(str.strip, all_scenes))

        for i, scene in enumerate(all_scenes):

            print("{}/{} {}:".format(i, len(all_scenes), scene), '------------------------')
            dist_thresh_secne = dist_thresh[scene]
            prob_thresh_secne = prob_thresh[scene]
            num_consist_secne = num_consist[scene]
            depths, projs, rgbs = load_data(args.root_path, args.depth_path, scene, prob_thresh_secne)
            tot_frame = depths.shape[0]
            height, width = depths.shape[2], depths.shape[3]
            points = []

            print('Scene: {} total: {} frames'.format(scene, tot_frame))
            for i in range(tot_frame):
                pc_buff = torch.zeros((3, height, width), device=depths.device, dtype=depths.dtype)
                val_cnt = torch.zeros((1, height, width), device=depths.device, dtype=depths.dtype)
                j = 0
                batch_size = 20

                while True:
                    ref_pc, pcs, dist = filter_depth(ref_depth=depths[i:i + 1], src_depths=depths[j:min(j + batch_size, tot_frame)],
                                                     ref_proj=projs[i:i + 1], src_projs=projs[j:min(j + batch_size, tot_frame)])
                    masks = (dist < dist_thresh_secne).float()
                    masked_pc = pcs * masks
                    pc_buff += masked_pc.sum(dim=0, keepdim=False)
                    val_cnt += masks.sum(dim=0, keepdim=False)

                    j += batch_size
                    if j >= tot_frame:
                        break

                final_mask = (val_cnt >= num_consist_secne).squeeze(0)
                avg_points = torch.div(pc_buff, val_cnt).permute(1, 2, 0)

                final_pc = extract_points(avg_points, final_mask, rgbs[i])
                points.append(final_pc)
                if i == 0 or i == tot_frame - 1:
                    print('Processing {} {}/{} ...'.format(scene, i + 1, tot_frame))

            ply_id = int(scene[4:])
            write_ply('{}/mvsnet{:03d}_13.ply'.format(args.ply_path, ply_id), np.concatenate(points, axis=0))
            del points, depths, rgbs, projs

            gc.collect()

            print('Save {}/mvsnet{:03d}_13.ply successful.'.format(args.ply_path, ply_id))


if __name__ == '__main__':
    dist_thresh = {
            'scan1': 0.1,
            'scan4': 0.1,
            'scan9': 0.1,
            'scan10': 0.1,
            'scan11': 0.1,
            'scan12': 0.1,
            'scan13': 0.1,
            'scan15': 0.1,
            'scan23': 0.1,
            'scan24': 0.1,
            'scan29': 0.1,
            'scan32': 0.1,
            'scan33': 0.1,
            'scan34': 0.1,
            'scan48': 0.1,
            'scan49': 0.1,
            'scan62': 0.1,
            'scan75': 0.1,
            'scan77': 0.08,
            'scan110': 0.1,
            'scan114': 0.1,
            'scan118': 0.1
        }

    prob_thresh = {
            'scan1': 0.3,
            'scan4': 0.3,
            'scan9': 0.3,
            'scan10': 0.3,
            'scan11': 0.3,
            'scan12': 0.3,
            'scan13': 0.3,
            'scan15': 0.3,
            'scan23': 0.3,
            'scan24': 0.3,
            'scan29': 0.3,
            'scan32': 0.3,
            'scan33': 0.3,
            'scan34': 0.3,
            'scan48': 0.3,
            'scan49': 0.3,
            'scan62': 0.3,
            'scan75': 0.3,
            'scan77': 0.2,
            'scan110': 0.3,
            'scan114': 0.3,
            'scan118': 0.3
        }

    num_consist = {
            'scan1': 4,
            'scan4': 3,
            'scan9': 3,
            'scan10': 4,
            'scan11': 3,
            'scan12': 3,
            'scan13': 4,
            'scan15': 3,
            'scan23': 3,
            'scan24': 3,
            'scan29': 3,
            'scan32': 3,
            'scan33': 3,
            'scan34': 3,
            'scan48': 3,
            'scan49': 3,
            'scan62': 3,
            'scan75': 4,
            'scan77': 4,
            'scan110': 3,
            'scan114': 3,
            'scan118': 3
        }
    open3d_filter(dist_thresh, prob_thresh, num_consist)