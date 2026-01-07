# -*- coding: utf-8 -*-

import torch

# ========= 辅助函数：RGB → HSV =========
def rgb_to_hsv(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    rgb: [B,3,H,W] 取值范围 [0,1]
    返回: [B,3,H,W]，分别是 H,S,V
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc, _ = torch.max(rgb, dim=1)
    minc, _ = torch.min(rgb, dim=1)
    v = maxc
    deltac = maxc - minc

    # Hue
    h = torch.zeros_like(maxc)
    mask = deltac > eps
    rmask = (maxc == r) & mask
    gmask = (maxc == g) & mask
    bmask = (maxc == b) & mask
    h[rmask] = ((g - b)[rmask] / deltac[rmask]) % 6
    h[gmask] = ((b - r)[gmask] / deltac[gmask]) + 2
    h[bmask] = ((r - g)[bmask] / deltac[bmask]) + 4
    h = h / 6  # 归一化到 [0,1]

    # Saturation
    s = deltac / (maxc + eps)

    hsv = torch.stack((h, s, v), dim=1)
    return hsv