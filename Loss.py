import torch
import torch.nn.functional as F
from Utils import rgb_to_hsv

def compute_channel_statistics(rgb, mask_gt, eps=1e-6):
    """
    计算真实mask区域内HSV通道的统计信息
    返回: (h_low_gt, h_high_gt, s_low_gt, s_high_gt, v_low_gt, v_high_gt)
    """
    # RGB -> HSV
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]  # [B,H,W]

    # 确保mask_gt是二值且与特征图相同尺寸
    if mask_gt.dim() == 4:
        mask_gt = mask_gt.squeeze(1)  # [B,1,H,W] -> [B,H,W]

    # 调整mask尺寸如果必要
    if mask_gt.shape[-2:] != h.shape[-2:]:
        mask_gt = F.interpolate(mask_gt.unsqueeze(1).float(),
                                size=h.shape[-2:],
                                mode='nearest').squeeze(1)

    batch_stats = []

    for i in range(rgb.shape[0]):  # 遍历batch
        current_mask = mask_gt[i] > 0.5  # 二值化
        if current_mask.sum() == 0:  # 如果没有前景区域
            # 使用默认值避免NaN
            batch_stats.append((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
            continue

        # 提取前景区域的HSV值
        h_fg = h[i][current_mask]
        s_fg = s[i][current_mask]
        v_fg = v[i][current_mask]

        # 计算分位数作为阈值区间（使用5%和95%分位数）
        h_low = torch.quantile(h_fg, 0.05)
        h_high = torch.quantile(h_fg, 0.95)
        s_low = torch.quantile(s_fg, 0.05)
        s_high = torch.quantile(s_fg, 0.95)
        v_low = torch.quantile(v_fg, 0.05)
        v_high = torch.quantile(v_fg, 0.95)

        batch_stats.append((h_low, h_high, s_low, s_high, v_low, v_high))

    # 堆叠成张量
    stats_tensor = torch.tensor(batch_stats, device=rgb.device)
    return (stats_tensor[:, 0], stats_tensor[:, 1],  # h_low, h_high
            stats_tensor[:, 2], stats_tensor[:, 3],  # s_low, s_high
            stats_tensor[:, 4], stats_tensor[:, 5])  # v_low, v_high


def hsv_unet_loss(logits, mask_gt, rgb_input, eps=1e-6):
    """
    改进的HSV UNet损失函数
    基于真实数据统计的阈值监督 + 基础分割损失
    """
    # 确保维度正确
    if mask_gt.dim() == 3:
        mask_gt = mask_gt.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]

    # 1) 基础分割损失
    if logits.shape[1] == 1:
        # 二分类情况
        seg_pred = torch.sigmoid(logits)
        loss_seg = F.binary_cross_entropy(seg_pred, mask_gt)
    else:
        # 多分类情况
        if mask_gt.shape[1] == 1:
            mask_gt_ce = mask_gt.squeeze(1).long()
        else:
            mask_gt_ce = mask_gt.long()
        loss_seg = F.cross_entropy(logits, mask_gt_ce)

    # 3) 总损失
    total_loss = loss_seg

    loss_dict = {
        'seg': round(loss_seg.item(), 4),
    }

    return total_loss, loss_dict