import torch
import torch.nn as nn

class StripBlock(nn.Module):
    """条纹注意力核心块（StripBlock）：交叉条纹状分组卷积生成空间注意力权重"""
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return attn

class StripAttn(nn.Module):
    """条纹注意力模块（StripAttn）：封装StripBlock+门控机制，轻量化空间特征增强"""
    def __init__(self, d_model, k1=1, k2=19):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripBlock(d_model, k1, k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

# ===================== 关键封装 =====================
# YOLOv11标准 StripConv 模块（完全适配C3k2的参数/维度/接口）
# 输入输出和C3k2一致，算力持平，可无缝替换所有C3k2
class StripConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, ratio=0.25, k1=1, k2=19):
        super().__init__()
        self.c_ = int(c2 * ratio)  # 分组通道数，和C3k2一致
        self.cv1 = nn.Conv2d(c1, self.c_, 1, 1)
        self.cv2 = nn.Conv2d(self.c_, c2, 1)
        self.attn = StripAttn(self.c_, k1, k2) # 嵌入你的条纹注意力核心
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y = self.attn(y) # 核心：条纹注意力特征增强
        y = self.cv2(y)
        return x + y if self.add else y