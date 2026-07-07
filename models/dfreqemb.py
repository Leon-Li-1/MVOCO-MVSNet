import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import cv2
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from scipy import ndimage
from .module import Conv2d

class InceptionConv2d(nn.Module):
    """ Inception depthweise convolution, 为了节约显存，这里仅仅使用nn.Conv3d
    """
    def __init__(self, in_channels, out_channels, square_kernel_size=3, band_kernel_size=11, groups=4):
        super().__init__()
        self.groups = groups
        gc = int(in_channels // groups)  # channel numbers of a convolution branch
        self.dwconv_d = nn.Conv2d(gc, gc, kernel_size=(square_kernel_size, square_kernel_size), padding=(square_kernel_size // 2, square_kernel_size // 2))
        self.dwconv_d1 = nn.Conv2d(gc, gc, kernel_size=(7, 7), padding=(7 // 2, 7 // 2))
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2))
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0))
        self.split_indexes = (gc, gc, gc, gc)
        self.conv2d = Conv2d(in_channels, out_channels, kernel_size=square_kernel_size, stride=1, padding=1)

    def forward(self, x):
        y = x
        # x_d0, x_w, x_h, x_d = torch.split(x, self.split_indexes, dim=1)
        x_d0, x_d, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x_d0 = self.dwconv_d(x_d0)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)
        x_d = self.dwconv_d1(x_d)
        x = y + torch.cat([x_d0, x_d, x_w, x_h], dim=1)  # 原始特征，提取后的高层次特征会不会进行互补呢？
        x = self.conv2d(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 1),
            InceptionConv2d(32, 32)
        )
        self.conv2 = InceptionConv2d(32, 32)
        self.conv3 = InceptionConv2d(32, 32)
        self.res = InceptionConv2d(32, 1)

    def forward(self, depth_init, img):
        concat = torch.cat([img, depth_init], dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class DepthEncoder(nn.Module):
    def __init__(self, depth_enc_dim=16, num_bands=8, num_coeffs=16):  # 这里最好的是8
        super().__init__()
        self.num_bands = num_bands
        self.num_coeffs = num_coeffs
        self.depth_encoder = nn.Sequential(
            # nn.Conv2d(1, 32, 3, padding=1),  # 直接融入无编码深度值
            # nn.Conv2d(4, 32, 3, padding=1),  # 无编码深度值 + 法线
            nn.Conv2d(self.num_bands * 2 + 3, 32, 3, padding=1),  # 深度编码+法线
            # nn.Conv2d(self.num_bands * 2 + self.num_coeffs + 3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, depth_enc_dim, 3, padding=1)
        )
        self.register_buffer('scharr_x', torch.tensor([[-3, 0, 3],
                                                     [-10, 0, 10],
                                                     [-3, 0, 3]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('scharr_y', torch.tensor([[3, 10, 3],
                                                     [0, 0, 0],
                                                     [-3, -10, -3]], dtype=torch.float32).view(1, 1, 3, 3))

    def compute_normals(self, depth):
        '''
        计算法线并计算改进的背景抑制损失
        :param depth: 深度图，形状为 B x 1 x H x W
        :param eps: 避免除零的小数
        :return: 法线图 B x 3 x H x W, 以及损失项
        '''
        B, _, H, W = depth.shape
        # 计算 梯度

        grad_x = F.conv2d(depth, self.scharr_x, padding=1)  # B x 1 x H x W
        grad_y = F.conv2d(depth, self.scharr_y, padding=1)  # B x 1 x H x W

        # **计算法向量**
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                                        torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")

        x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        p1 = torch.cat([x_grid, y_grid, depth], dim=1)
        p2 = torch.cat([x_grid + 1, y_grid, depth + grad_x], dim=1)
        p3 = torch.cat([x_grid, y_grid + 1, depth + grad_y], dim=1)

        normals = torch.cross(p2 - p1, p3 - p1, dim=1)
        normals = F.normalize(normals, p=2, dim=1)

        return normals

    def compute_normals_real(self, depth, K):
        """
        计算真实相机坐标系下的法线
        :param depth: [B, 1, H, W]
        :param K:     [B, 4, 4] 或 [B, 3, 3]
        """
        B, _, H, W = depth.shape
        device = depth.device

        # --- 1. 安全提取内参并调整维度 ---
        # 确保提取出的参数形状为 [B, 1, 1]，以便与 [B, 1, H, W] 的 depth 广播
        # 注意：这里取 K[:, 0, 0] 得到的是 [B]，必须 view 成 [B, 1, 1]
        fx = K[:, 0, 0].view(B, 1, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1, 1)

        # --- 2. 构建带 Batch 维度的网格 ---
        # 生成 [H, W]
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        # 扩展为 [1, 1, H, W] -> 再 expand 为 [B, 1, H, W]
        # 这一步至关重要，保证后续运算都在 B 维度上对齐
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        y_coords = y_coords.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        # --- 3. 反投影 (Backprojection) ---
        # 此时所有变量维度均为 [B, 1, H, W]，运算结果也是 [B, 1, H, W]
        X = (x_coords - cx) * depth / fx
        Y = (y_coords - cy) * depth / fy
        Z = depth  # Z 本身就是 [B, 1, H, W]

        # --- 4. 计算梯度 ---
        # Scharr 算子卷积后保持维度不变 (padding=1 保持大小)
        # 结果维度: [B, 1, H, W]
        grad_X_u = F.conv2d(X, self.scharr_x, padding=1)
        grad_Y_u = F.conv2d(Y, self.scharr_x, padding=1)
        grad_Z_u = F.conv2d(Z, self.scharr_x, padding=1)

        grad_X_v = F.conv2d(X, self.scharr_y, padding=1)
        grad_Y_v = F.conv2d(Y, self.scharr_y, padding=1)
        grad_Z_v = F.conv2d(Z, self.scharr_y, padding=1)

        # --- 5. 组合切向量并计算法线 ---
        # cat dim=1 后维度变为 [B, 3, H, W]
        tangent_u = torch.cat([grad_X_u, grad_Y_u, grad_Z_u], dim=1)
        tangent_v = torch.cat([grad_X_v, grad_Y_v, grad_Z_v], dim=1)

        normals = torch.cross(tangent_u, tangent_v, dim=1)
        normals = F.normalize(normals, p=2, dim=1)

        return normals

    def geometric_position_encoding(self, depth_map, num_bands=8):
        """
        将深度值映射到高维空间，编码几何位置信息
        depth_map: [B, 1, H, W] 深度图
        num_bands: 频带数量
        """
        # 向量化生成所有频率
        freqs = 2.0 ** torch.arange(num_bands, device=depth_map.device).float()

        # 调整维度用于广播 [B, 1, H, W] × [num_bands] → [B, num_bands, H, W]
        phase = freqs.view(1, num_bands, 1, 1) * math.pi * depth_map

        # 一次性计算所有频带的正弦和余弦编码
        sin_enc = torch.sin(phase)
        cos_enc = torch.cos(phase)

        # 交错排列正弦和余弦编码
        encodings = torch.stack([sin_enc, cos_enc], dim=2)  # [B, num_bands, 2, H, W]
        encodings = encodings.view(depth_map.size(0), 2 * num_bands, depth_map.size(2), depth_map.size(3))

        return encodings

    def local_frequency_encoding(self, depth_map, window_size=8, num_coeffs=16):
        """
        使用DCT变换提取局部频率特征
        window_size: 窗口大小
        num_coeffs: 保留的低频系数数量
        """
        B, C, H, W = depth_map.shape
        
        # 确保窗口大小不超过图像尺寸
        window_size = min(window_size, H, W)
        
        # 使用unfold提取图像块
        patches = depth_map.unfold(2, window_size, window_size // 2).unfold(3, window_size, window_size // 2)
        
        # 调整维度: [B, C, num_patches_h, num_patches_w, window_size, window_size]
        patches = patches.contiguous().view(B, C, -1, window_size, window_size)
        
        # 应用2D DCT - 使用自定义DCT实现
        dct_coeffs = self._dct_2d(patches)
        
        # 计算要保留的低频系数数量
        low_freq_size = int(math.sqrt(num_coeffs))
        low_freq_size = min(low_freq_size, window_size)
        
        # 保留低频分量
        coeffs = dct_coeffs[..., :low_freq_size, :low_freq_size]

        # 重新调整维度
        num_patches_h = (H - window_size) // (window_size // 2) + 1
        num_patches_w = (W - window_size) // (window_size // 2) + 1

        coeffs_reshaped = coeffs.reshape(B, C, num_patches_h, num_patches_w, low_freq_size * low_freq_size)
        coeffs_reshaped = coeffs_reshaped.permute(0, 1, 4, 2, 3).contiguous()
        
        # 重采样回原始分辨率
        encoding = F.interpolate(
            coeffs_reshaped.view(B, C * low_freq_size * low_freq_size, num_patches_h, num_patches_w), 
            size=(H, W), mode='bilinear', align_corners=False
        )

        # 调整输出维度
        output = encoding.view(B, C, low_freq_size * low_freq_size, H, W)
        return output.squeeze(1)  # [B, num_coeffs, H, W]
    
    def _dct_1d(self, x):
        """1D DCT实现"""
        N = x.size(-1)
        
        # 构建DCT矩阵
        n = torch.arange(N, device=x.device).float()
        k = torch.arange(N, device=x.device).float().unsqueeze(1)
        
        dct_matrix = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix *= math.sqrt(2 / N)
        dct_matrix[0] *= math.sqrt(1 / 2)  # 第一个系数的归一化
        
        return torch.matmul(x, dct_matrix.T)
    
    def _dct_2d(self, x):
        """2D DCT实现"""
        # 对最后一个维度应用DCT
        x_dct = self._dct_1d(x)
        
        # 对倒数第二个维度应用DCT
        x_dct = x_dct.transpose(-1, -2)
        x_dct = self._dct_1d(x_dct)
        x_dct = x_dct.transpose(-1, -2)
        
        return x_dct
    
    def _idct_1d(self, x):
        """1D逆DCT实现"""
        N = x.size(-1)
        
        # 构建逆DCT矩阵
        k = torch.arange(N, device=x.device).float()
        n = torch.arange(N, device=x.device).float().unsqueeze(1)
        
        idct_matrix = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        idct_matrix *= math.sqrt(2 / N)
        idct_matrix[:, 0] *= math.sqrt(1 / 2)  # 第一个系数的归一化
        
        return torch.matmul(x, idct_matrix)

    def forward(self, depth_map, intrins=None):
        if intrins is None:
            min_depth = depth_map.min()
            max_depth = depth_map.max()
            # 归一化深度值
            normalized_depth = (depth_map - min_depth) / (max_depth - min_depth + 1e-8)
            # 深度信息编码
            geom_enc = self.geometric_position_encoding(normalized_depth, self.num_bands)
            # geo_local_enc = self.local_frequency_encoding(normalized_depth, self.num_coeffs)
            surf_normal = self.compute_normals(normalized_depth)  # 法线计算
            # depth_enc = torch.cat([geom_enc, geo_local_enc, surf_normal], dim=1)
        else:
            # 深度信息编码
            geom_enc = self.geometric_position_encoding(depth_map, self.num_bands)
            # geo_local_enc = self.local_frequency_encoding(normalized_depth, self.num_coeffs)
            surf_normal = self.compute_normals_real(depth_map, intrins)
            # depth_enc = torch.cat([geom_enc, geo_local_enc, surf_normal], dim=1)
        depth_enc = torch.cat([geom_enc, surf_normal], dim=1)  # 完整的深度编码+法线
        # depth_enc = torch.cat([normalized_depth, surf_normal], dim=1)  # 深度值+法线
        # depth_enc = self.depth_encoder(normalized_depth)  # 只融入深度值
        depth_enc = self.depth_encoder(depth_enc)
        return depth_enc


class RayEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(RayEncoder, self).__init__()

    def get_pixel_coords(self, h, w, device):
        # 使用PyTorch张量操作替代NumPy，直接在目标设备上创建
        pixel_coords = torch.ones((3, h, w), dtype=torch.float32, device=device)
        
        # 创建坐标网格
        x_range = torch.arange(w, dtype=torch.float32, device=device).reshape(1, w).repeat(h, 1) + 0.5
        y_range = torch.arange(h, dtype=torch.float32, device=device).reshape(h, 1).repeat(1, w) + 0.5
        
        pixel_coords[0, :, :] = x_range
        pixel_coords[1, :, :] = y_range
        
        return pixel_coords.unsqueeze(0)  # (1, 3, H, W)

    def get_ray(self, intrins, H, W, return_uv=False):
        B, _, _ = intrins.shape
        device = intrins.device
        
        # 提取相机内参并调整维度用于广播
        fu = intrins[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
        cu = intrins[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
        fv = intrins[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
        cv = intrins[:, 1, 2].unsqueeze(-1).unsqueeze(-1)
        
        # 获取像素坐标并确保在相同设备上
        pixel_coords = self.get_pixel_coords(H, W, device)
        
        # 裁剪到所需尺寸并复制批次
        ray = pixel_coords[:, :, :H, :W].repeat(B, 1, 1, 1)
        
        # 计算归一化设备坐标
        ray[:, 0, :, :] = (ray[:, 0, :, :] - cu) / fu
        ray[:, 1, :, :] = (ray[:, 1, :, :] - cv) / fv
        
        if return_uv:
            return ray[:, :2, :, :]  # 返回UV坐标
        else:
            # 归一化得到单位方向向量
            ray = F.normalize(ray, dim=1)
            return ray

    def forward(self, intrins, H, W):
        return self.get_ray(intrins, H, W, return_uv=False)
