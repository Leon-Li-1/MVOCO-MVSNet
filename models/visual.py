import numpy as np  # NOQA
import matplotlib.pyplot as plt
import os
import torch  # NOQA
from skimage.feature import canny


def visualize_and_save_depth_maps(depth1, depth2, save_path=None):
    """
    显示并保存refine前后深度图及差异图
    参数:
        depth1: refine前深度图 (B 1 H W 格式的torch tensor)
        depth2: refine后深度图 (B 1 H W 格式的torch tensor)
        save_path: 保存路径（可选）
    """
    # 转换为numpy数组
    depth1_np = depth1.squeeze().detach().cpu().numpy()
    depth2_np = depth2.squeeze().detach().cpu().numpy()

    # 验证形状
    if depth1_np.ndim != 2 or depth2_np.ndim != 2:
        raise ValueError("输入深度图必须是2D数组")

    # 计算差异图（refine后 - refine前）
    diff_map = depth2_np - depth1_np
    abs_diff = np.abs(diff_map)  # 绝对差异

    # 创建3列对比图（原始/优化/差异）
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    # 原始深度图
    im1 = ax1.imshow(depth1_np, cmap='viridis')
    ax1.set_title('Raw Depth')
    # fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.axis('off')
    # 优化后深度图
    im2 = ax2.imshow(depth2_np, cmap='viridis')
    ax2.set_title('Refined Depth')
    # fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    plt.axis('off')
    # 差异图（使用发散色图突出正负差异）
    im3 = ax3.imshow(diff_map, cmap='coolwarm', vmin=-np.max(abs_diff), vmax=np.max(abs_diff))
    ax3.set_title('Difference (Refined - Raw)')
    # fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def visualize_normal_map(normal_tensor, save_path=None):
    """
    法向量可视化函数
    参数:
        normal_tensor: 法向量图 (B 3 H W 格式的torch tensor)
        save_path: 可选保存路径
    """
    # 转换并处理法向量数据
    normal_np = normal_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    normal_np = (normal_np + 1) / 2  # 将法向量从[-1,1]映射到[0,1]

    # 创建可视化窗口
    plt.figure(figsize=(8, 8))
    plt.imshow(normal_np)
    plt.title('Normal Map Visualization')
    plt.axis('off')

    # 保存功能
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"法向量图已保存至: {save_path}")

    plt.show()


def visualize_improvements(raw, refined, save_path=None):
    """几何特征增强对比可视化工具"""
    # 数据准备
    raw = raw.squeeze().detach().cpu().numpy()
    refined = refined.squeeze().detach().cpu().numpy()

    # 计算特征指标
    def compute_metrics(feat):
        l2_norm = np.linalg.norm(feat, axis=0)
        edge_map = np.sum([canny(channel) for channel in feat], axis=0)
        texture_strength = np.std(feat, axis=0)
        return l2_norm, edge_map, texture_strength

    # 创建可视化面板
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3)

    # 原始特征可视化
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.mean(raw, axis=0), cmap='cividis')
    ax1.set_title('Original Features')
    plt.axis('off')

    # 改进特征可视化
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.mean(refined, axis=0), cmap='cividis')
    ax2.set_title('Enhanced Features')
    plt.axis('off')

    # 改进差异热力图
    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.mean(refined, axis=0) - np.mean(raw, axis=0)
    im3 = ax3.imshow(diff, cmap='bwr', vmin=-abs(diff).max(), vmax=abs(diff).max())
    # fig.colorbar(im3, ax=ax3)
    ax3.set_title('Feature Enhancement Map')
    plt.axis('off')

    # # 边缘特征对比
    # _, raw_edges, _ = compute_metrics(raw)
    # _, refined_edges, _ = compute_metrics(refined)
    # ax4 = fig.add_subplot(gs[1, 0])
    # ax4.imshow(raw_edges, cmap='gray')
    # ax4.set_title('Original Edge Features')
    # plt.axis('off')

    # ax5 = fig.add_subplot(gs[1, 1])
    # ax5.imshow(refined_edges, cmap='gray')
    # ax5.set_title('Enhanced Edge Features')
    # plt.axis('off')

    # ax6 = fig.add_subplot(gs[1, 2])
    # ax6.imshow(refined_edges - raw_edges, cmap='coolwarm')
    # ax6.set_title('Edge Improvement')
    # plt.axis('off')

    # # 纹理强度对比
    # _, _, raw_texture = compute_metrics(raw)
    # _, _, refined_texture = compute_metrics(refined)
    # ax7 = fig.add_subplot(gs[2, 0])
    # ax7.imshow(raw_texture, cmap='cividis')
    # ax7.set_title('Original Texture Strength')
    # plt.axis('off')

    # ax8 = fig.add_subplot(gs[2, 1])
    # ax8.imshow(refined_texture, cmap='cividis')
    # ax8.set_title('Enhanced Texture Strength')
    # plt.axis('off')

    # ax9 = fig.add_subplot(gs[2, 2])
    # texture_diff = refined_texture - raw_texture
    # im9 = ax9.imshow(texture_diff, cmap='coolwarm', vmin=-abs(texture_diff).max(),
    #                  vmax=abs(texture_diff).max())
    # # fig.colorbar(im9, ax=ax9)
    # ax9.set_title('Texture Compensation')
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
