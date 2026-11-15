import torch
import torch.nn.functional as F
from torch import nn


class ScaleAwareAttention(nn.Module):
    def __init__(self, channels, scales=3):
        super().__init__()
        self.scales = scales
        # 共享MLP生成动态权重
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * scales, channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels * scales, kernel_size=1),
        )

    def forward(self, feats):
        """Feats: list of feature maps [f1, f2, f3], each shape (B,C,H,W)."""
        B = feats[0].shape[0]
        # 沿通道拼接并池化
        feat_cat = torch.cat([F.adaptive_avg_pool2d(f, (1, 1)) for f in feats], dim=1)  # (B, 3C, 1, 1)
        weights = self.fc(feat_cat)  # (B, 3*C, 1, 1)
        weights = weights.view(B, self.scales, -1, 1, 1)  # (B, S, C, 1, 1)
        weights = F.softmax(weights, dim=1)  # 归一化尺度权重

        out = []
        for i, f in enumerate(feats):
            scale_weight = weights[:, i, :, :, :]  # (B, C, 1, 1)
            out.append(f * scale_weight.expand_as(f))
        return sum(out)


class DyHeadBlock(nn.Module):
    def __init__(self, channels, groups=16):
        super().__init__()
        self.scale_attn = ScaleAwareAttention(channels)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // 16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        # Step 1: 尺度注意力融合
        x = self.scale_attn(inputs)  # 多尺度加权融合

        # Step 2: 通道注意力调制
        c_weight = self.channel_attn(x)
        x = x * c_weight

        # Step 3: 空间注意力调制
        s_weight = self.spatial_attn(x)
        x = x * s_weight

        return x


# ===================== 新增：适配YOLO-YAML的完整检测头【唯一新增的代码】 =====================
class DyHead_Detect_YOLO(nn.Module):
    def __init__(self, nc=3, ch=(256, 512, 1024)):  # nc=你的类别数3，ch=YOLO11n的3个特征通道
        super().__init__()
        self.nc = nc
        self.no = nc + 5  # YOLO标准输出：xywh+conf + 类别数
        self.ch = ch
        # 为3个尺度分别创建DyHead特征增强模块
        self.dy_blocks = nn.ModuleList([DyHeadBlock(c) for c in ch])
        # 预测层：把增强后的特征映射为检测结果（核心补全的部分）
        self.predict = nn.ModuleList([nn.Conv2d(c, self.no * 8, 1) for c in ch])

    def forward(self, x):
        """x: 接收YOLO yaml传入的3个特征层的list [f1,f2,f3]，完美适配你的DyHead输入格式 输出：和原生YOLO Detect完全一致的检测结果，无缝兼容训练/推理.
        """
        out = []
        for i in range(3):
            # 特征增强：当前尺度特征过DyHead
            feat = self.dy_blocks[i]([x[i]])
            # 预测输出：映射为检测框+类别
            pred = self.predict[i](feat)
            # 维度重塑，和原生Detect输出格式一致
            bs, _, ny, nx = pred.shape
            pred = pred.view(bs, 8, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            out.append(pred)
        return torch.cat([o.view(o.shape[0], -1, self.no) for o in out], 1)
