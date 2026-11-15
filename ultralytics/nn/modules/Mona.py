# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ------------------------------ Mona 模块 修正无错完整版 ------------------------------
# # 适配YOLOv11 | 纯PyTorch无MMCV依赖 | 即插即用 | 轻量级多尺度特征增强 | 航拍小目标专用
# INNER_DIM = 64

# class MonaOp(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()
#         # 多尺度深度可分离卷积：3+5+7核，覆盖小/中/大目标特征，无参数量爆炸
#         self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
#         self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
#         self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
#         self.projector = nn.Conv2d(in_features, in_features, kernel_size=1) # 1x1卷积特征聚合

#     def forward(self, x):
#         identity = x
#         # 多尺度特征融合+均值加权+残差连接，梯度平滑不爆炸
#         conv1_x = self.conv1(x)
#         conv2_x = self.conv2(x)
#         conv3_x = self.conv3(x)
#         x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

#         identity = x
#         x = self.projector(x)
#         return identity + x # 双重残差，特征增强不丢失

# class Mona(nn.Module): # ✅ 核心修正：删除MMCV的BaseModule，改为原生nn.Module
#     def __init__(self, in_dim, factor=4):
#         super().__init__()
#         self.in_dim = in_dim
#         # 特征投影层：降维→增强→升维，轻量级设计
#         self.project1 = nn.Linear(in_dim, INNER_DIM)
#         self.nonlinear = nn.GELU() # ✅ 优化：改为Module类，比F.gelu更规范，可被优化器识别
#         self.project2 = nn.Linear(INNER_DIM, in_dim)
#         self.dropout = nn.Dropout(p=0.1) # 轻微dropout防止过拟合，航拍小目标数据集适配
#         self.adapter_conv = MonaOp(INNER_DIM) # 核心多尺度卷积增强
#         self.norm = nn.LayerNorm(in_dim) # 层归一化，梯度稳定
#         # 可学习缩放因子，初始化极小值，不破坏原始特征，训练中自适应增强
#         self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
#         self.gammax = nn.Parameter(torch.ones(in_dim))

#     def forward(self, x, hw_shape=None): # ✅ 修正参数名：hw_shapes → hw_shape，匹配调用逻辑
#         identity = x
#         # 双权重归一化增强，核心创新点，不丢失原始特征
#         x = self.norm(x) * self.gamma + x * self.gammax

#         # 特征投影降维
#         project1 = self.project1(x)
#         b, n, c = project1.shape

#         # 序列特征 → 特征图 [b,n,c] → [b,c,h,w]，适配卷积运算
#         h, w = hw_shape
#         project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
#         project1 = self.adapter_conv(project1) # 多尺度卷积增强
#         # 特征图 → 序列特征，恢复原始维度
#         project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

#         # 非线性激活+dropout+升维投影+残差连接
#         nonlinear = self.nonlinear(project1)
#         nonlinear = self.dropout(nonlinear)
#         project2 = self.project2(nonlinear)

#         return identity + project2 # 最终残差，特征增强无副作用

import torch
from torch import nn

# ------------------------------ 纯卷积版 Mona 模块 (唯一适配替换SPPF) ------------------------------
# 输入: [B, C, H, W] 特征图 | 输出: [B, C, H, W] 特征图 | 尺寸不变、通道不变 | 无维度转换 | 无额外参数
INNER_DIM = 64


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # 3/5/7核多尺度深度可分离卷积，覆盖航拍小/中/大目标，无参数量爆炸
        self.conv1 = nn.Conv2d(in_features, in_features, 3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, 5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, 7, padding=3, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, 1, 1, 0)

    def forward(self, x):
        identity = x
        x = (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3.0 + identity
        return identity + self.projector(x)


class Mona(nn.Module):
    def __init__(self, in_dim, factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, INNER_DIM, 1, 1, 0)  # 1x1卷积降维，轻量
        self.nonlinear = nn.GELU()
        self.conv2 = nn.Conv2d(INNER_DIM, in_dim, 1, 1, 0)  # 1x1卷积升维，恢复原通道
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = MonaOp(INNER_DIM)
        self.norm = nn.GroupNorm(1, in_dim)  # 适配卷积特征图的归一化，梯度稳
        # 可学习权重，初始化极小值，不破坏原始特征，训练自适应增强
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))

    def forward(self, x):
        identity = x
        # 双权重归一化+残差增强，特征提纯不丢失
        x = self.norm(x) * self.gamma + x * self.gammax
        # 纯卷积运算，无任何维度转换/重塑，速度极快
        x = self.conv1(x)
        x = self.adapter_conv(x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return identity + x
