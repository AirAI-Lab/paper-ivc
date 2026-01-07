# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import rgb_to_hsv

# ========= 可微直方图 ===软分配，每个像素点，给等分的每个n_bins区间的中心点都产生贡献======
class DifferentiableHistogram(nn.Module):
    def __init__(self, n_bins=32, v_min=0., v_max=1., sigma=0.02):
        """
        n_bins: 直方图bin数量
        v_min, v_max: 亮度范围
        sigma: 高斯宽度，控制平滑度
        """
        super().__init__()
        self.register_buffer("centers", torch.linspace(v_min, v_max, n_bins))
        self.sigma = sigma

    def forward(self, v):
        """
        v: [B,1,H,W] 亮度
        return: [B, n_bins]
        """
        B = v.shape[0]
        v_flat = v.view(B, 1, -1)  # [B,1,N]
        dist = (v_flat - self.centers.view(1, -1, 1)) ** 2
        weights = torch.exp(-dist / (2 * self.sigma ** 2))
        hist = weights.mean(-1)  # [B, bins]
        return hist


# ========= HSV注意力机制 =========
class HSV_Attention(nn.Module):
    """
    HSV注意力机制：
      - 输入: RGB 图像 [B,3,H,W], 归一化到 [0,1]
      - 输出: attention_mask ∈ [0,1], shape [B,1,H,W]
              以及预测的H通道阈值和V通道阈值
    """

    def __init__(self, bins=32, alpha_init=50.0):
        super().__init__()
        self.hist_h = DifferentiableHistogram(n_bins=bins)  # H通道直方图
        self.hist_s = DifferentiableHistogram(n_bins=bins)  # S通道直方图
        self.hist_v = DifferentiableHistogram(n_bins=bins)  # V通道直方图

        # MLP for S channel thresholds
        self.mlp_h = nn.Sequential(
            nn.Linear(bins, 128),  # 增加容量
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),  # 逐步降维
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # 输出2个值
            nn.Sigmoid()  # 约束到[0,1]范围
        )

        # MLP for S channel thresholds
        self.mlp_s = nn.Sequential(
            nn.Linear(bins, 128),  # 增加容量
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),  # 逐步降维
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # 输出2个值
            nn.Sigmoid()  # 约束到[0,1]范围
        )

        # MLP for V channel thresholds
        self.mlp_v = nn.Sequential(
            nn.Linear(bins, 128),  # 增加容量
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),  # 逐步降维
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # 输出2个值
            nn.Sigmoid()  # 约束到[0,1]范围
        )

        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # 平滑卷积
        self.smooth = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=False
        )
        with torch.no_grad():
            self.smooth.weight[:] = 1.0 / 9.0  # 均值滤波
        for p in self.smooth.parameters():
            p.requires_grad = False

    def forward(self, rgb):
        # 1. RGB -> HSV
        hsv = rgb_to_hsv(rgb)
        h = hsv[:, 0:1, :, :]  # H通道 [B,1,H,W]
        s = hsv[:, 1:2, :, :]  # S通道 [B,1,H,W]
        v = hsv[:, 2:3, :, :]  # V通道 [B,1,H,W]

        # 2. 分别计算S和V通道的直方图特征
        hist_feat_h = self.hist_h(h)  # [B, bins]
        hist_feat_s = self.hist_s(s)  # [B, bins]
        hist_feat_v = self.hist_v(v)  # [B, bins]

        # 3. MLP 预测S和V通道的阈值
        # H通道阈值
        t_h = self.mlp_h(hist_feat_h)  # [B,2]
        t_h_sorted, _ = torch.sort(t_h, dim=1)  # 保证 low < high
        h_low, h_high = t_h_sorted[:, 0], t_h_sorted[:, 1]

        # S通道阈值
        t_s = self.mlp_s(hist_feat_s)  # [B,2]
        t_s_sorted, _ = torch.sort(t_s, dim=1)  # 保证 low < high
        s_low, s_high = t_s_sorted[:, 0], t_s_sorted[:, 1]

        # V通道阈值
        t_v = self.mlp_v(hist_feat_v)  # [B,2]
        t_v_sorted, _ = torch.sort(t_v, dim=1)  # 保证 low < high
        v_low, v_high = t_v_sorted[:, 0], t_v_sorted[:, 1]

        # 4. 分别生成H,S和V通道的soft mask
        # H通道mask
        m_h_low = torch.sigmoid(self.alpha * (h - h_low.view(-1, 1, 1, 1)))
        m_h_high = 1 - torch.sigmoid(self.alpha * (h - h_high.view(-1, 1, 1, 1)))
        mask_h = m_h_low * m_h_high  # [B,1,H,W]

        # S通道mask
        m_s_low = torch.sigmoid(self.alpha * (s - s_low.view(-1, 1, 1, 1)))
        m_s_high = 1 - torch.sigmoid(self.alpha * (s - s_high.view(-1, 1, 1, 1)))
        mask_s = m_s_low * m_s_high  # [B,1,H,W]

        # V通道mask
        m_v_low = torch.sigmoid(self.alpha * (v - v_low.view(-1, 1, 1, 1)))
        m_v_high = 1 - torch.sigmoid(self.alpha * (v - v_high.view(-1, 1, 1, 1)))
        mask_v = m_v_low * m_v_high  # [B,1,H,W]

        # 5. 融合H和V通道的mask（使用逻辑与操作，即相乘）
        mask = mask_h * mask_s * mask_v  # [B,1,H,W]
        mask = self.smooth(mask)  # 平滑输出

        return mask, (h_low, h_high, s_low, s_high, v_low, v_high)


# ========= UNet主干网络组件 =========
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ========= 普通UNet网络（无HSV指导） =========
class Plain_UNet(nn.Module):
    """
    普通UNet网络，无HSV注意力指导
    输入: RGB图像 [B,3,H,W]
    输出: 分割结果 [B,num_classes,H,W]
    保持与HSV_Guided_UNet相同的调用方式和输出格式
    """

    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(Plain_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # UNet编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # UNet解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 1. UNet编码器路径
        x1 = self.inc(x)  # [B,64,H,W]
        x2 = self.down1(x1)  # [B,128,H/2,W/2]
        x3 = self.down2(x2)  # [B,256,H/4,W/4]
        x4 = self.down3(x3)  # [B,512,H/8,W/8]
        x5 = self.down4(x4)  # [B,1024,H/16,W/16]

        # 2. UNet解码器路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 3. 输出层
        logits = self.outc(x)

        # 4. 为了保持与HSV_Guided_UNet相同的输出格式
        # 返回全1的注意力掩码和零值的阈值
        batch_size, _, h, w = x.shape
        attention_mask = torch.ones(batch_size, 1, h, w, device=x.device)

        # 创建与batch_size匹配的零值阈值 (为了直接适配代码)
        h_low = torch.zeros(batch_size, device=x.device)
        h_high = torch.zeros(batch_size, device=x.device)
        s_low = torch.zeros(batch_size, device=x.device)
        s_high = torch.zeros(batch_size, device=x.device)
        v_low = torch.zeros(batch_size, device=x.device)
        v_high = torch.zeros(batch_size, device=x.device)

        return logits, attention_mask, (h_low, h_high, s_low, s_high, v_low, v_high)


# ========= 多尺度HSV注意力引导UNet =========
class MultiScale_HSV_UNet(nn.Module):
    """
    多尺度HSV注意力引导UNet
    在不同尺度上应用HSV注意力
    """

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, attention_bins=32):
        super(MultiScale_HSV_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 多尺度HSV注意力
        self.hsv_attention = HSV_Attention(bins=attention_bins)

        # 多尺度注意力调整层
        self.attention_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            ) for _ in range(4)
        ])

        # UNet编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # UNet解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 新增残差融合模块
        self.residual_fusion = nn.Sequential(
            nn.Conv2d(64 + 3, 64, kernel_size=3, padding=1),  # 融合64通道特征+3通道RGB
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()  # 门控机制
        )

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 保存原始输入用于残差连接
        original_x = x  # [B,3,H,W]

        # 1. 获取基础HSV注意力掩码
        base_attention, thresholds = self.hsv_attention(x)

        # 2. 生成多尺度注意力图
        attention_maps = []
        current_att = base_attention
        for i in range(4):
            # 下采样注意力图
            if i > 0:
                current_att = F.avg_pool2d(current_att, kernel_size=2, stride=2)
            # 调整注意力图
            adjusted_att = self.attention_pyramid[i](current_att)
            attention_maps.append(adjusted_att)

        # 3. UNet编码器路径 + 多尺度注意力
        x1 = self.inc(x)
        x1_att = x1 * attention_maps[0]  # 原始尺度

        x2 = self.down1(x1_att)
        x2_att = x2 * attention_maps[1]  # 1/2尺度

        x3 = self.down2(x2_att)
        x3_att = x3 * attention_maps[2]  # 1/4尺度

        x4 = self.down3(x3_att)
        x4_att = x4 * attention_maps[3]  # 1/8尺度

        x5 = self.down4(x4_att)

        # 4. UNet解码器路径
        x = self.up1(x5, x4_att)
        x = self.up2(x, x3_att)
        x = self.up3(x, x2_att)
        x = self.up4(x, x1_att)

        # 5. 残差融合 - 核心改进
        # 调整原始输入尺寸（如果需要）
        if original_x.shape[-2:] != x.shape[-2:]:
            original_resized = F.interpolate(original_x, size=x.shape[-2:],
                                             mode='bilinear', align_corners=False)
        else:
            original_resized = original_x

        # 门控残差融合
        fusion_input = torch.cat([x, original_resized], dim=1)  # [B,67,H,W]
        gate_weights = self.residual_fusion(fusion_input)  # [B,64,H,W]

        # 残差连接：门控加权融合
        fused_features = x * gate_weights + original_resized.mean(1, keepdim=True).expand_as(x) * (1 - gate_weights)

        # 6. 输出层
        logits = self.outc(fused_features)

        return logits, base_attention, thresholds


# ========= 注意力引导的UNet网络（修正版） =========
class HSV_Guided_UNet(nn.Module):
    """
    基于HSV注意力引导的UNet网络 - 修正版
    使用真正的注意力机制：注意力图与特征图相乘
    """

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, attention_bins=32):
        super(HSV_Guided_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # HSV注意力机制
        self.hsv_attention = HSV_Attention(bins=attention_bins)

        # UNet编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # UNet解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 注意力权重调整层（可选）
        self.attention_adjust1 = nn.Conv2d(1, 1, kernel_size=1)
        self.attention_adjust2 = nn.Conv2d(1, 1, kernel_size=1)
        self.attention_adjust3 = nn.Conv2d(1, 1, kernel_size=1)
        self.attention_adjust4 = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        # 1. 获取HSV注意力掩码
        attention_mask, thresholds = self.hsv_attention(x)

        # 2. UNet编码器路径 + 真正的注意力机制
        x1 = self.inc(x)  # [B,64,H,W]

        # 注意力机制：调整注意力图并与特征图相乘
        att1 = F.interpolate(attention_mask, size=x1.shape[2:], mode='bilinear', align_corners=False)
        att1 = self.attention_adjust1(att1)  # 可学习的调整
        att1 = torch.sigmoid(att1)  # 确保在0-1范围内
        x1_att = x1 * att1  # 真正的注意力：特征图 * 注意力图

        x2 = self.down1(x1_att)  # [B,128,H/2,W/2]

        # 第二层注意力
        att2 = F.interpolate(attention_mask, size=x2.shape[2:], mode='bilinear', align_corners=False)
        att2 = self.attention_adjust2(att2)
        att2 = torch.sigmoid(att2)
        x2_att = x2 * att2

        x3 = self.down2(x2_att)  # [B,256,H/4,W/4]

        # 第三层注意力
        att3 = F.interpolate(attention_mask, size=x3.shape[2:], mode='bilinear', align_corners=False)
        att3 = self.attention_adjust3(att3)
        att3 = torch.sigmoid(att3)
        x3_att = x3 * att3

        x4 = self.down3(x3_att)  # [B,512,H/8,W/8]

        # 第四层注意力
        att4 = F.interpolate(attention_mask, size=x4.shape[2:], mode='bilinear', align_corners=False)
        att4 = self.attention_adjust4(att4)
        att4 = torch.sigmoid(att4)
        x4_att = x4 * att4

        x5 = self.down4(x4_att)  # [B,1024,H/16,W/16]

        # 3. UNet解码器路径
        x = self.up1(x5, x4_att)
        x = self.up2(x, x3_att)
        x = self.up3(x, x2_att)
        x = self.up4(x, x1_att)

        # 4. 输出层
        logits = self.outc(x)

        return logits, attention_mask, thresholds



# ========= 注意力引导的UNet网络 =========
class HSV_Guided_UNet_ChannelConcat(nn.Module):
    """
    基于HSV注意力引导的UNet网络
    输入: RGB图像 [B,3,H,W]
    输出: 分割结果 [B,num_classes,H,W]
    """

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, attention_bins=32):
        super(HSV_Guided_UNet_ChannelConcat, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # HSV注意力机制
        self.hsv_attention = HSV_Attention(bins=attention_bins)

        # UNet编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # UNet解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 注意力融合层
        self.attention_fusion1 = nn.Conv2d(65, 64, kernel_size=1)  # 64+1=65
        self.attention_fusion2 = nn.Conv2d(129, 128, kernel_size=1)  # 128+1=129
        self.attention_fusion3 = nn.Conv2d(257, 256, kernel_size=1)  # 256+1=257
        self.attention_fusion4 = nn.Conv2d(513, 512, kernel_size=1)  # 512+1=513

    def forward(self, x):
        # 1. 获取HSV注意力掩码
        attention_mask, thresholds = self.hsv_attention(x)

        # 2. UNet编码器路径
        x1 = self.inc(x)  # [B,64,H,W]

        # 融合注意力到第一层特征
        attention_resized = F.interpolate(attention_mask, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x1_att = torch.cat([x1, attention_resized], dim=1)  # [B,65,H,W]
        x1_fused = self.attention_fusion1(x1_att)  # [B,64,H,W]

        x2 = self.down1(x1_fused)  # [B,128,H/2,W/2]

        # 融合注意力到第二层特征
        attention_resized = F.interpolate(attention_mask, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x2_att = torch.cat([x2, attention_resized], dim=1)  # [B,129,H/2,W/2]
        x2_fused = self.attention_fusion2(x2_att)  # [B,128,H/2,W/2]

        x3 = self.down2(x2_fused)  # [B,256,H/4,W/4]

        # 融合注意力到第三层特征
        attention_resized = F.interpolate(attention_mask, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x3_att = torch.cat([x3, attention_resized], dim=1)  # [B,257,H/4,W/4]
        x3_fused = self.attention_fusion3(x3_att)  # [B,256,H/4,W/4]

        x4 = self.down3(x3_fused)  # [B,512,H/8,W/8]

        # 融合注意力到第四层特征
        attention_resized = F.interpolate(attention_mask, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x4_att = torch.cat([x4, attention_resized], dim=1)  # [B,513,H/8,W/8]
        x4_fused = self.attention_fusion4(x4_att)  # [B,512,H/8,W/8]

        x5 = self.down4(x4_fused)  # [B,1024,H/16,W/16]

        # 3. UNet解码器路径
        x = self.up1(x5, x4_fused)
        x = self.up2(x, x3_fused)
        x = self.up3(x, x2_fused)
        x = self.up4(x, x1_fused)

        # 4. 输出层
        logits = self.outc(x)

        return logits, attention_mask, thresholds


# ========= 简化版本（如果计算资源有限） =========
class Simplified_HSV_UNet(nn.Module):
    """
    简化的HSV引导UNet，只在编码器开始阶段使用注意力
    """

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, attention_bins=32):
        super(Simplified_HSV_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # HSV注意力机制
        self.hsv_attention = HSV_Attention(bins=attention_bins)

        # UNet编码器
        self.inc = DoubleConv(n_channels + 1, 64)  # 输入通道：RGB + 注意力掩码
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # UNet解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 1. 获取HSV注意力掩码
        attention_mask, thresholds = self.hsv_attention(x)

        # 2. 将注意力掩码与输入图像拼接
        x_with_attention = torch.cat([x, attention_mask], dim=1)  # [B,4,H,W]

        # 3. 标准的UNet前向传播
        x1 = self.inc(x_with_attention)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits, attention_mask, thresholds


# ========= 测试代码 =========
if __name__ == "__main__":
    # 测试网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建测试数据
    batch_size, channels, height, width = 2, 3, 256, 256
    x = torch.randn(batch_size, channels, height, width).to(device)

    print("Testing Plain_UNet...")
    model_plain = MultiScale_HSV_UNet(n_channels=3, n_classes=1).to(device)
    output_plain, attention_mask_plain, thresholds_plain = model_plain(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_plain.shape}")
    print(f"Attention mask shape: {attention_mask_plain.shape}")
    print(f"Thresholds: {[(t[0].item(), t[1].item()) for t in zip(*thresholds_plain)]}")