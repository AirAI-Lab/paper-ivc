import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class HSV_Attention(nn.Module):
    """
    修正后的 HSV 多尺度注意力模块
    输出与 UNet 编码器通道/尺度一一对应：
      att_64  -> [B,64,H,W]
      att_128 -> [B,128,H/2,W/2]
      att_256 -> [B,256,H/4,W/4]
      att_512 -> [B,512,H/8,W/8]
    """
    def __init__(self, bins=32, chs=[64, 128, 256, 512]):
        super(HSV_Attention, self).__init__()
        self.bins = bins
        self.chs = chs
        self.eps = 1e-6

        # 单通道 -> bins 的 1x1 卷积（可以视作 soft-binning / 学习映射）
        self.h_conv = nn.Conv2d(1, self.bins, kernel_size=1, bias=False)
        self.s_conv = nn.Conv2d(1, self.bins, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(1, self.bins, kernel_size=1, bias=False)

        # 为每个编码器尺度生成对应通道数的注意力
        self.scale_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.bins * 3, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                nn.Sigmoid()
            ) for ch in self.chs
        ])

        self.pool = nn.AvgPool2d(2, stride=2)

    def rgb2hsv_torch(self, x):
        """
        输入 [B,3,H,W] 值在 [0,1]，输出 HSV，各通道归一化：
          H -> [0,1] (原来为 0..2pi)
          S -> [0,1]
          V -> [0,1]
        """
        r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        max_rgb = torch.max(x, dim=1)[0]
        min_rgb = torch.min(x, dim=1)[0]
        delta = max_rgb - min_rgb

        h = torch.zeros_like(max_rgb)
        mask = delta > self.eps
        # 计算色相（0..6），再映射到 0..1
        h_val = torch.where(max_rgb == r,
                            (g - b) / (delta + self.eps),
                            torch.where(max_rgb == g,
                                        (b - r) / (delta + self.eps) + 2,
                                        (r - g) / (delta + self.eps) + 4))
        h[mask] = h_val[mask]
        # h 目前在 [0,6)，映射到 [0,1]
        h = h / 6.0

        s = delta / (max_rgb + self.eps)
        v = max_rgb

        hsv = torch.stack([h, s, v], dim=1)
        return hsv

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = x / 255.0 if x.max() > 1 else x
        hsv = self.rgb2hsv_torch(x_norm)
        h, s, v = hsv[:, 0:1, :, :], hsv[:, 1:2, :, :], hsv[:, 2:3, :, :]

        h_feat = self.h_conv(h)
        s_feat = self.s_conv(s)
        v_feat = self.v_conv(v)
        feat = torch.cat([h_feat, s_feat, v_feat], dim=1)  # [B, bins*3, H, W]

        # 按 UNet 编码尺度依次下采样并生成对应通道的注意力
        att_64 = self.scale_conv[0](feat)                 # [B,64,H,W]
        feat_1 = self.pool(feat)
        att_128 = self.scale_conv[1](feat_1)              # [B,128,H/2,W/2]
        feat_2 = self.pool(feat_1)
        att_256 = self.scale_conv[2](feat_2)              # [B,256,H/4,W/4]
        feat_3 = self.pool(feat_2)
        att_512 = self.scale_conv[3](feat_3)              # [B,512,H/8,W/8]

        return att_64, att_128, att_256, att_512



class DoubleConv(nn.Module):
    """UNet基础双卷积块：Conv-BN-ReLU × 2"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """UNet下采样块：MaxPool + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """UNet上采样块：上采样 + 拼接 + 双卷积"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        # 双线性插值上采样 或 转置卷积上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """UNet输出层：1×1卷积映射到类别数"""
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MultiScale_HSV_UNet(nn.Module):
    """
    多尺度HSV注意力引导UNet
    在不同尺度上应用HSV注意力增强特征
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, attention_bins=32):
        super(MultiScale_HSV_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # 定义UNet各尺度的通道数
        self.encoder_chs = [64, 128, 256, 512]
        # 多尺度HSV注意力模块（输出通道数与编码器层匹配）
        self.hsv_attention = HSV_Attention(bins=attention_bins, chs=self.encoder_chs)

        # UNet编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # UNet解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 获取多尺度HSV注意力权重（64,128,256,512）
        att_64, att_128, att_256, att_512 = self.hsv_attention(x)

        # 2. UNet编码器路径 + 多尺度注意力特征增强
        # python
        # 替换 MultiScale_HSV_UNet.forward 中注意力应用的片段
        # 假设 att_64, att_128, att_256, att_512 已从 self.hsv_attention(x) 得到

        # 第一层：64 通道，匹配 att_64
        x1 = self.inc(x)  # [B,64,H,W]
        att_64_resize = F.interpolate(att_64, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x1_att = x1 * att_64_resize  # 注意力增强

        # 第二层：128 通道，匹配 att_128
        x2 = self.down1(x1_att)  # [B,128,H/2,W/2]
        att_128_resize = F.interpolate(att_128, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x2_att = x2 * att_128_resize

        # 第三层：256 通道，匹配 att_256
        x3 = self.down2(x2_att)  # [B,256,H/4,W/4]
        att_256_resize = F.interpolate(att_256, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x3_att = x3 * att_256_resize

        # 第四层：512 通道，匹配 att_512
        x4 = self.down3(x3_att)  # [B,512,H/8,W/8]
        att_512_resize = F.interpolate(att_512, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x4_att = x4 * att_512_resize

        # 编码器最后一层
        x5 = self.down4(x4_att)  # [B,1024/factor,H/16,W/16]

        # 3. UNet解码器路径
        x = self.up1(x5, x4_att)
        x = self.up2(x, x3_att)
        x = self.up3(x, x2_att)
        x = self.up4(x, x1_att)

        # 4. 输出层
        logits = self.outc(x)

        return logits


# 测试代码
if __name__ == "__main__":
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构建模型
    model = MultiScale_HSV_UNet(n_channels=3, n_classes=1, bilinear=True, attention_bins=32).to(device)
    # 生成测试输入 [B,3,H,W]
    x = torch.randn(2, 3, 512, 512).to(device)
    # 前向传播
    output = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")