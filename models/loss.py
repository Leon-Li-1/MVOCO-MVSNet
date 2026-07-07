# -*- coding: utf-8 -*-
# @Description: Loss Functions (Sec 3.4 in the paper).
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import torch
import torch.nn.functional as F
import torch.nn as nn
from models.tools import *


class MVSLoss(nn.Module):
    def __init__(self, args):
        super(MVSLoss, self).__init__()
        # self.loss_funcs = [SupLossMultiStage(args)]  # CasMVSNet训练实验
        self.loss_funcs = [UnsupLossMultiStage_l05(args), SupLossMultiStage(args)]  # CasMVSNet训练实验
        self.args = args

    def forward(self, data, outputs):
        losses = {}
        scalar_outputs = {}
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=data["imgs"][0].device, requires_grad=False)
        for loss_func in self.loss_funcs:
            loss, scalar_outputs = loss_func(data, outputs)
            losses[loss_func.name] = loss.item()
            total_loss = total_loss + loss
        return total_loss, scalar_outputs


class UnSupLoss(nn.Module):
    def __init__(self, args):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()
        self.args = args

    def forward(self, imgs, cams, depth, stage_idx):
        # imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        num_views = len(imgs)
        ref_img = imgs[0]
        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.125)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.25)
        elif stage_idx == 2:
            ref_img = F.interpolate(ref_img, scale_factor=0.5)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0
        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.125)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.25)
            elif stage_idx == 2:
                view_img = F.interpolate(view_img, scale_factor=0.5)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            if mask.sum() == 0:
                self.unsup_loss = torch.tensor(0.0, dtype=torch.float32, device=mask.device)
                return self.unsup_loss
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss_l0_5(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)
            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))
        # #smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = self.args.wrecon * self.reconstr_loss + self.args.wssim * self.ssim_loss + self.args.wsmooth * self.smooth_loss
        return self.unsup_loss


class UnsupLossMultiStage_l05(nn.Module):
    def __init__(self, args):
        super(UnsupLossMultiStage_l05, self).__init__()
        self.name = "unslossl05"
        self.args = args
        self.unsup_loss = UnSupLoss(args)

    def forward(self, data, outputs, **kwargs):
        inputs = outputs
        imgs = data["center_imgs"]
        cams = data["proj_matrices"]
        depth_loss_weights = self.args.stage_lw
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs[0].device, requires_grad=False)
        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1
        # if stage_idx > 0:
            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)
            if depth_loss_weights is not None:
                total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss = total_loss + 1.0 * depth_loss
            # scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            # scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            # scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            # scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss
        return total_loss, scalar_outputs


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        y = y.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        SSIM_mask = self.mask_pool(mask)

        output = SSIM_mask * (torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1) + 1e-6)

        if torch.sum(SSIM_mask.type(torch.float32)) == 0:
            output = torch.zeros_like(output, dtype=torch.float32, device=output.device)
        return output.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]


class SupLossMultiStage(nn.Module):
    def __init__(self, args):
        super(SupLossMultiStage, self).__init__()
        self.name = "suploss"
        self.args = args

    # def smooth_loss(self, depth_est, depth_gt, mask):
    #     mask = mask > 0.5
    #     return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
    
    def entropy_loss(self, prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
        # from AA
        mask_true = mask
        valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6

        shape = depth_gt.shape          # B,H,W

        depth_num = depth_value.shape[1]
        if len(depth_value.shape) < 3:
            depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2, 3, 0, 1)     # B,N,H,W
        else:
            depth_value_mat = depth_value

        gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)

        gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
        gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W

        # gt index map -> gt one hot volume (B x 1 x H x W )
        gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

        # cross entropy image (B x D X H x W)
        cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1)  # B, 1, H, W

        # masked cross entropy loss
        masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)  # valid pixel
        masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

        masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)  # Origin use sum : aggregate with batch
        # winner-take-all depth map
        wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
        wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

        if return_prob_map:
            photometric_confidence = torch.max(prob_volume, dim=1)[0]  # output shape dimension B * H * W
            return masked_cross_entropy, wta_depth_map, photometric_confidence
        return masked_cross_entropy, wta_depth_map

    def mvsnet_loss(self, inputs, depth_gt_ms, mask_ms, depth_values, **kwargs):
        depth_min, depth_max = depth_values[:, 0], depth_values[:, -1]
        depth_loss_weights = self.args.stage_lw
        depth_refine_loss_weights = self.args.refine_stage_lw

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
        total_entropy = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            prob_volume = stage_inputs["prob_volume"]
            depth_value = stage_inputs["depth_hypo"]
            depth_gt = depth_gt_ms[stage_key]
            mask = mask_ms[stage_key]
            mask = mask > 0.5
            if self.args.mask_rate != 0.0:
                keep_prob = 1.0 - self.args.mask_rate
                # 仅在 mask==True 的位置做伯努利采样，False位置保持为False
                keep_mask = torch.bernoulli(
                    torch.full_like(mask, keep_prob, dtype=torch.float32)
                ).bool()
                mask = mask & keep_mask
            entropy_weight = 2.0
            entro_loss, depth_entropy = self.entropy_loss(prob_volume, depth_gt, mask, depth_value)  # Transformer说可以增强细节实现
            entro_loss = entro_loss * entropy_weight
            depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
            total_entropy += entro_loss
            stage_idx = int(stage_key.replace("stage", "")) - 1
            if stage_idx == 0:
                total_loss += depth_loss_weights[stage_idx] * entro_loss * self.args.wsup
            else:
                # 中间阶段的距离损失
                if self.args.which_module == 'mvoco_plus':
                    depth_refine = stage_inputs["depth_refine"]
                    depth_refine_loss = F.smooth_l1_loss(depth_refine[mask], depth_gt[mask], reduction='mean')  # smooth_l1_loss
                    total_loss = total_loss + depth_loss_weights[stage_idx] * entro_loss * self.args.wsup + self.args.wrefine * depth_refine_loss_weights[stage_idx - 1] * depth_refine_loss
                elif self.args.which_module == 'mvoco':
                    total_loss = total_loss + depth_loss_weights[stage_idx] * entro_loss * self.args.wsup

        depth_pred = stage_inputs['depth']
        depth_gt = depth_gt_ms[stage_key]
        epe = cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max)
        return total_loss, epe, depth_loss, total_entropy, entro_loss

    # def mvsnet_loss(self, inputs, depth_gt_ms, mask_ms):
    #     depth_loss_weights = self.args.stage_lw
    #     total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    #     for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
    #         depth_values = stage_inputs["depth"]
    #         depth_gt = depth_gt_ms[stage_key]
    #         mask = mask_ms[stage_key]
    #         mask = mask > 0.5
    #         depth_loss = self.smooth_loss(depth_values, depth_gt, mask)
    #         if depth_loss_weights is not None:
    #             stage_idx = int(stage_key.replace("stage", "")) - 1
    #             # 计算深度精炼网络的深度损失
    #             if stage_idx > 0:
    #                 depth_refine = stage_inputs["refined_depth"]
    #                 depth_refine_loss = self.smooth_loss(depth_refine, depth_gt, mask)
    #             else:
    #                 depth_refine_loss = 0.0
    #             total_loss += depth_loss_weights[stage_idx] * depth_loss * self.args.wsup + 0.5 * depth_refine_loss  # 0.5
    #         else:
    #             total_loss += depth_loss

    #     return total_loss, depth_loss

    def forward(self, data, outputs, **kwargs):
        inputs = outputs
        scalar_outputs = {}
        depth_gt_ms = data["depth"]
        mask_ms = data["mask"]
        depth_values = data["depth_values"]
        total_loss, epe, depth_loss, total_entropy, entro_loss= self.mvsnet_loss(inputs, depth_gt_ms, mask_ms, depth_values)
        scalar_outputs["depth_loss"] = depth_loss
        scalar_outputs["epe"] = epe
        scalar_outputs["total_entropy"] = total_entropy
        scalar_outputs["entro_loss"] = entro_loss
        return total_loss, scalar_outputs


# def entropy_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
#     # from AA
#     mask_true = mask
#     valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6

#     shape = depth_gt.shape          # B,H,W

#     depth_num = depth_value.shape[1]
#     if len(depth_value.shape) < 3:
#         depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2, 3, 0, 1)     # B,N,H,W
#     else:
#         depth_value_mat = depth_value

#     gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)

#     gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
#     gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W

#     # gt index map -> gt one hot volume (B x 1 x H x W )
#     gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

#     # cross entropy image (B x D X H x W)
#     cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1)  # B, 1, H, W

#     # masked cross entropy loss
#     masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)  # valid pixel
#     masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

#     masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)  # Origin use sum : aggregate with batch
#     # winner-take-all depth map
#     wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
#     wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

#     if return_prob_map:
#         photometric_confidence = torch.max(prob_volume, dim=1)[0]  # output shape dimension B * H * W
#         return masked_cross_entropy, wta_depth_map, photometric_confidence
#     return masked_cross_entropy, wta_depth_map


# def mambamvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
#     depth_values = kwargs.get("depth_values")
#     depth_min, depth_max = depth_values[:, 0], depth_values[:, -1]
#     with_normal_loss = True
#     depth_loss_weights = kwargs.get("stage_lw", [1, 1, 1, 1])
#     depth_refine_loss_weights = kwargs.get("dstage_lw", [0.25, 0.5, 1.0])

#     total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#     total_entropy = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#     for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
#         prob_volume = stage_inputs["prob_volume"]
#         depth_value = stage_inputs["depth_hypo"]
#         depth_gt = depth_gt_ms[stage_key]
#         mask = mask_ms[stage_key]
#         mask = mask > 0.5
#         entropy_weight = 2.0

#         # entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_value)
#         entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_value)  # Transformer说可以增强细节实现
#         entro_loss = entro_loss * entropy_weight
#         depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
#         total_entropy += entro_loss
#         stage_idx = int(stage_key.replace("stage", "")) - 1
#         if stage_idx == 0:
#             total_loss += depth_loss_weights[stage_idx] * entro_loss
#         else:
#             # total_loss = total_loss + depth_loss_weights[stage_idx] * entro_loss
#             # 中间阶段的距离损失
#             depth_refine = stage_inputs["depth_refine"]
#             # depth_refine_prob = stage_inputs['depth_refine_prob']
#             depth_refine_loss = F.smooth_l1_loss(depth_refine[mask], depth_gt[mask], reduction='mean')  # smooth_l1_loss
#             total_loss = total_loss + depth_loss_weights[stage_idx] * entro_loss + 0.1 * depth_refine_loss_weights[stage_idx - 1] * depth_refine_loss
#             # total_loss = total_loss + beta * (0.8 * depth_loss_weights[stage_idx] * entro_loss + 0.2 * depth_loss_weights[stage_idx] * js_loss) + (1 - beta) * depth_refine_loss_weights[stage_idx - 1] * depth_refine_loss

#     depth_pred = stage_inputs['depth']
#     depth_gt = depth_gt_ms[stage_key]
#     epe = cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max)
#     return total_loss, epe, depth_loss, total_entropy, entro_loss


def cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max):
    depth_pred_norm = depth_pred * 128 / (depth_max - depth_min)[:, None, None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:, None, None]

    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    epe = abs_err.mean()
    err1 = (abs_err <= 1).float().mean() * 100
    err3 = (abs_err <= 3).float().mean() * 100

    return epe  # err1, err3
