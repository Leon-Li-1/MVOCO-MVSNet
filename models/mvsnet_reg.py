import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # NOQA
from models.submodules_reg import *
import matplotlib.pyplot as plt

class StageNet(nn.Module):
    def __init__(self, args, attn_temp=2):
        super(StageNet, self).__init__()
        self.attn_temp = attn_temp
        self.which_dataset = args.which_dataset

    def forward(self, features, proj_matrices, depth_hypo, regnet, group_cor_dim, depth_interal_ratio):

        # @Note step1: feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_feature.shape[1]

        # @Note step2: cost aggregation
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
        cor_weight_sum = 1e-8
        cor_feats = 0
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            save_fn = None
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)

            warped_src = warped_src.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
            ref_volume = ref_volume.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
            cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W 
            del warped_src, src_proj, src_fea

            cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W 
            cor_weight_sum += cor_weight  # B D H W
            cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
            del cor_weight, cor_feat

        cost_volume = cor_feats / cor_weight_sum.unsqueeze(1)  # B N C D H W -> B C D H W 
        del cor_weight_sum, src_features

        # @Note step3: cost regularization
        cost_reg = regnet(cost_volume)
        del cost_volume

        prob_volume = F.softmax(cost_reg, dim=1)  # B D H W
        #  @Note step4: depth regression
        depth = depth_regression(prob_volume, depth_hypotheses=depth_hypo)  # (b, h, w)
        num_depth = prob_volume.shape[1]
        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1,
                                                padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume,
                                           depth_hypotheses=torch.arange(num_depth, device=prob_volume.device,
                                                                         dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            pv = torch.where(prob_volume <= 0, torch.ones_like(prob_volume) * 1e-5, prob_volume)
            distribution_consistency = (np.log(pv.shape[1]) - torch.sum(-pv * torch.log(pv), dim=1)) / np.log(pv.shape[1])
        # 深度转换
        last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
        inverse_min_depth = 1 / depth + depth_interal_ratio * last_depth_itv  # B H W
        inverse_max_depth = 1 / depth - depth_interal_ratio * last_depth_itv  # B H W

        output_stage = {
            "depth": depth,
            "photometric_confidence": photometric_confidence,
            "depth_hypo": depth_hypo,
            "prob_volume": prob_volume,
            "inverse_min_depth": inverse_min_depth,
            "inverse_max_depth": inverse_max_depth,
            # "depth_reg": depth_reg
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
        self.StageNet = StageNet(args)

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
            B, C, H, W = features[0][stage_name].shape
            proj_matrices_stage = proj_matrices[stage_name]

            ref_img_stage = F.interpolate(imgs[0], size=None, scale_factor=1. / 2**(3 - stage_idx), mode="bilinear", align_corners=False)
            features_stage = [feat[stage_name] for feat in features]
            # @Note features
            if self.which_module == 'mvoco_plus':
                if stage_idx >= 1:
                    depth_last = F.interpolate(depth_last.unsqueeze(1), size=None, scale_factor=2, mode="bilinear", align_corners=False)
                    features_stage[0], refined_depth = self.depth_aware_ff[stage_idx - 1](features_stage[0], depth_last, ref_img_stage, proj_matrices_stage[:, 0, 1])

            # @Note depth hypos
            if stage_idx == 0:
                depth_hypo = init_inverse_range(depth_values, self.hypo_plane_num_stages[stage_idx], img[0].device, img[0].dtype, H, W)
            else:
                inverse_min_depth, inverse_max_depth = outputs_stage['inverse_min_depth'].detach(), outputs_stage['inverse_max_depth'].detach()
                depth_hypo = schedule_inverse_range(inverse_min_depth, inverse_max_depth, self.hypo_plane_num_stages[stage_idx], H, W)  # B D H W

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
            outputs.update(outputs_stage)

        return outputs



class MVSNet(nn.Module):
    def __init__(self, args):
        super(MVSNet, self).__init__()
        self.model = BaseMVSNet(args)

    def forward(self, data, mode):
        assert mode in ["train", "val", "test"], "mode wrong!"
        outputs = self.model(data, mode)
        return outputs
