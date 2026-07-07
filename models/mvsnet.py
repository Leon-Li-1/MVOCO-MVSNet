# -*- coding: utf-8 -*-
# @Description: Main network architecture for GeoMVSNet.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # NOQA
from models.submodules import *  # NOQA 根据实验内容,手动调整使用 submodules_test 或者 submodules
import matplotlib.pyplot as plt  # NOQA


class StageNet(nn.Module):
    def __init__(self, attn_temp=2):
        super(StageNet, self).__init__()
        self.attn_temp = attn_temp

    def forward(self, features, proj_matrices, depth_hypo, regnet, group_cor_dim, depth_interal_ratio):

        # @Note step1: feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)  # 两个矩阵，内参和外参
        ref_feature, src_features = features[0], features[1:]  # 参考图和源图
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]  # 对应的投影矩阵分开
        B, D, H, W = depth_hypo.shape
        C = ref_feature.shape[1]

        # @Note step2: cost aggregation
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)  # 参考图D方向每个深度图上聚合
        cor_weight_sum = 1e-8
        cor_feats = 0
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):  # 同时迭代两个迭代对象的方法，r和s图片两两做操作
            save_fn = None
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])  # 计算源图K·Mext的值，1对应内参，2对应外参
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])  # 计算参考图K·Mext的值
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W

            warped_src = warped_src.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)  # 按照channel通道，把数据分成多个group，类似于多头注意力机制这样的
            ref_volume = ref_volume.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
            cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W  这里简单的相乘相当于计算了关系度，获得的是相关性组特征
            del warped_src, src_proj, src_fea

            cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W   其实这里类似self-attention的操作
            cor_weight_sum += cor_weight  # B D H W
            cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
            del cor_weight, cor_feat

        cost_volume = cor_feats / cor_weight_sum.unsqueeze(1)  # B N C D H W -> B C D H W  # 这里聚合的方法使用的是grounp聚合
        del cor_weight_sum, src_features

        # @Note step3: cost regularization
        cost_reg = regnet(cost_volume)
        del cost_volume

        prob_volume = F.softmax(cost_reg, dim=1)  # B D H W

        #  @Note step4: depth regression
        #  赢者通吃
        prob_max_indices = prob_volume.max(1, keepdim=True)[1]  # B 1 H W
        depth = torch.gather(depth_hypo, 1, prob_max_indices).squeeze(1)  # B H W，这里深度去概率最大的地方，并不是取的期望值
        with torch.no_grad():
            photometric_confidence = prob_volume.max(1)[0]  # B H W，prob_volume.max的返回值是两部分的
            photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1), scale_factor=1, mode='bilinear', align_corners=True).squeeze(1)  # 这里为什么要经过1倍的插值进行平滑一次

        # 深度转换
        last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]  # 上一次的深度假设间距
        inverse_min_depth = 1 / depth + depth_interal_ratio * last_depth_itv  # B H W 按照提前设定好的比例缩小深度假设的比例，这次生成的深度±深度假设间距
        inverse_max_depth = 1 / depth - depth_interal_ratio * last_depth_itv  # B H W

        output_stage = {
            "depth": depth,
            "photometric_confidence": photometric_confidence,
            "depth_hypo": depth_hypo,
            "prob_volume": prob_volume,
            "inverse_min_depth": inverse_min_depth,
            "inverse_max_depth": inverse_max_depth,
            # "photometric_confidence_1": photometric_confidence_1,
        }
        return output_stage


class BaseMVSNet(nn.Module):
    def __init__(self, args):
        super(BaseMVSNet, self).__init__()
        self.which_module = args.which_module
        self.levels = args.levels
        self.hypo_plane_num_stages = [int(n) for n in args.hypo_plane_num_stages.split(",")]
        self.depth_interal_ratio_stages = [float(ir) for ir in args.depth_interal_ratio_stages.split(",")]
        self.feat_base_channel = args.feat_base_channel
        self.reg_base_channel = args.reg_base_channel
        self.group_cor_dim_stages = [int(n) for n in args.group_cor_dim_stages.split(",")]
        self.StageNet = StageNet()

        # feature settings
        self.FeatureNet = FPNFeature(self.feat_base_channel)
        if self.which_module == 'mvoco_plus':
            self.depth_aware_ff = nn.ModuleList([DepthAwareFF(args, feat_dim=32), DepthAwareFF(args, feat_dim=16), DepthAwareFF(args, feat_dim=8)])
        # cost regularization settings
        self.RegNet_stages = nn.ModuleList()
        for stage_idx in range(self.levels):
            in_dim = self.group_cor_dim_stages[stage_idx]
            self.RegNet_stages.append(UNet3DCNNReg(input_channel=in_dim, base_channel=self.reg_base_channel))

    def forward(self, sample_cuda, mode):

        outputs = {}
        imgs = sample_cuda["imgs"]
        proj_matrices = sample_cuda["proj_matrices"]
        depth_values = sample_cuda["depth_values"]

        features = []
        for nview_idx in range(len(imgs)):
            img = imgs[nview_idx]
            features.append(self.FeatureNet(img))

        # coarse-to-fine
        for stage_idx in range(self.levels):
            stage_name = "stage{}".format(stage_idx + 1)
            B, C, H, W = features[0][stage_name].shape  # 只有一维，里边存的是第几张图片，每一个维度存的却是每个图片的四个stage的特征
            proj_matrices_stage = proj_matrices[stage_name]

            ref_img_stage = F.interpolate(imgs[0], size=None, scale_factor=1. / 2**(3 - stage_idx), mode="bilinear", align_corners=False)  # 通过插值的方式，将图片裁剪出来
            features_stage = [feat[stage_name] for feat in features]
            # @Note features
            if self.which_module == 'mvoco_plus':
                if stage_idx >= 1:
                    depth_last = F.interpolate(depth_last.unsqueeze(1), size=None, scale_factor=2, mode="bilinear", align_corners=False)  # 对生成的深度信息也要进行上采样
                    features_stage[0], refined_depth = self.depth_aware_ff[stage_idx - 1](features_stage[0], depth_last, ref_img_stage, proj_matrices_stage[:, 0, 1])

            # @Note depth hypos
            if stage_idx == 0:
                depth_hypo = init_inverse_range(depth_values, self.hypo_plane_num_stages[stage_idx], img[0].device, img[0].dtype, H, W)
            else:
                inverse_min_depth, inverse_max_depth = outputs_stage['inverse_min_depth'].detach(), outputs_stage['inverse_max_depth'].detach()
                depth_hypo = schedule_inverse_range(inverse_min_depth, inverse_max_depth, self.hypo_plane_num_stages[stage_idx], H, W)  # B D H W 根据上次深度图获取这次深度图的假设

            # @Note cost regularization
            outputs_stage = self.StageNet(
                features_stage, proj_matrices_stage, depth_hypo=depth_hypo,
                regnet=self.RegNet_stages[stage_idx], group_cor_dim=self.group_cor_dim_stages[stage_idx],
                depth_interal_ratio=self.depth_interal_ratio_stages[stage_idx]
            )
            # 存储refine值
            if self.which_module == 'mvoco_plus':
                if stage_idx >= 1:
                    outputs_stage['depth_refine'] = refined_depth.squeeze(1)

                depth_last = outputs_stage['depth']
            outputs[stage_name] = outputs_stage
            outputs.update(outputs_stage)  # 这里使用update而不使用append是一个小细节

        return outputs


class MVSNet(nn.Module):
    def __init__(self, args):
        super(MVSNet, self).__init__()
        self.model = BaseMVSNet(args)

    def forward(self, data, mode):
        assert mode in ["train", "val", "test"], "mode wrong!"
        outputs = self.model(data, mode)
        return outputs
