import torch
import torch.nn.functional as F
from torch import nn


class DropPath(nn.Module):
    """Per-sample stochastic depth without requiring an external dependency."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


# 简化工具函数（保留核心）
def to_3d(x):
    return x.flatten(2).transpose(1, 2)


def to_4d(x, h, w):
    b, _, c = x.shape
    return x.transpose(1, 2).reshape(b, c, h, w)


# 适配YOLO的轻量化MLP（统一SiLU激活）
class YOLOMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 0.5)  # 隐藏层减半，降低复杂度
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()  # 统一YOLO的激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 修复数值稳定性的LinearAttention
class StableLinearAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 确保head_dim是整数（避免维度不匹配）
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.lepe = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # 深度卷积
            nn.BatchNorm2d(dim),  # 添加BN，稳定特征分布
        )
        # 缩放因子（避免数值过度缩小）
        self.scale = self.head_dim**-0.5

    def forward(self, x, x_shape):
        b, n, c = x.shape
        h, w = x_shape[2:]

        # 拆分qkv并调整形状（更稳定的写法）
        qk = self.qk(x).reshape(b, n, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # [b, num_heads, n, head_dim]
        v = x.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 数值稳定的注意力计算（删除ELU，避免分布偏移）
        q = q * self.scale  # 合理缩放，避免数值爆炸
        k = k * self.scale

        # 简化注意力计算（无分母风险）
        attn = F.softmax(q @ k.transpose(-2, -1), dim=-1)  # 用softmax替代ELU+除法，更稳定
        x_out = (attn @ v).transpose(1, 2).reshape(b, n, c)

        # LePE分支（添加BN后更稳定）
        v_4d = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [b, c, h, w]
        lepe = self.lepe(v_4d).permute(0, 2, 3, 1).reshape(b, n, c)

        return x_out + lepe  # 残差融合LePE


# 核心：轻量化LAMA（适配YOLO11n）
class LAMA(nn.Module):
    def __init__(self, dim, num_heads=4, drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 简化CPE层（添加BN，稳定特征）
        self.cpe = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.BatchNorm2d(dim), nn.SiLU())

        # 核心注意力层
        self.attn = StableLinearAttention(dim=dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim)

        # 轻量化MLP
        self.mlp = YOLOMLP(in_features=dim, drop=drop)
        self.norm2 = nn.LayerNorm(dim)

        # 残差路径的dropout（轻量化）
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        _B, _C, H, W = x.shape
        H * W
        x_origin = x  # 保存原始输入，用于全局残差

        # 步骤1：CPE特征增强（仅1层，无冗余）
        x_cpe = self.cpe(x).flatten(2).permute(0, 2, 1)  # [b, n, c]

        # 步骤2：注意力增强（核心）
        x_3d = to_3d(x)  # [b, n, c]
        x_3d = self.norm1(x_3d + x_cpe)  # 局部残差
        x_3d = self.attn(x_3d, x.shape)

        # 步骤3：轻量化MLP
        x_3d = x_3d + self.drop_path(self.mlp(self.norm2(x_3d)))

        # 步骤4：转回4D并添加全局残差（最关键！保留YOLO原有特征）
        x_4d = to_4d(x_3d, H, W)
        return x_4d + x_origin  # 全局残差：LAMA增强特征 + YOLO原始特征


# 测试代码（验证模块有效性）
if __name__ == "__main__":
    lama = LAMA(dim=128, num_heads=4)
    x = torch.randn(4, 128, 128, 128)  # YOLO11n P4分支输入
    out = lama(x)
    print(f"LAMA输入形状: {x.shape}")  # torch.Size([4, 128, 128, 128])
    print(f"LAMA输出形状: {out.shape}")  # torch.Size([4, 128, 128, 128])
    print("模块输出维度匹配，无报错！")
