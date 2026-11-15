import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelPool(nn.Module):
    """通道池化模块：拼接全局最大池化与全局平均池化结果，增强特征表达"""
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1),
             torch.mean(x, 1).unsqueeze(1)),
            dim=1
        )


class Basic(nn.Module):
    """基础卷积模块：封装“卷积+批归一化+激活函数”的通用结构"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CALayer(nn.Module):
    """通道注意力层（CA）：结合行列维度池化与激发操作，增强通道相关性"""
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))
        self.conv_1x1 = nn.Conv2d(
            in_channels=2 * channel,
            out_channels=2 * channel,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(2 * channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.LeakyReLU()
        self.F_w = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.F_h = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        N, C, H, W = x.size()
        res = x
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))
        x_h, x_w = x.split(C, 1)
        x_h = self.F_h(x_h)
        x_w = self.F_w(x_w)
        s_h = self.sigmoid(x_h).expand_as(res)
        s_w = self.sigmoid(x_w).expand_as(res)
        out = res * s_h * s_w
        return out


class spatial_attn_layer(nn.Module):
    """空间注意力层（SA）：基于通道池化与卷积生成空间注意力权重"""
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Basic(
            in_planes=2,
            out_planes=1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bn=False,
            relu=False
        )
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class RCSSC(nn.Module):
    """残差通道-空间-尺度组合模块（RCSSC）：适配YOLOv11 C3k2替换 【仅修改此处，核心逻辑不变】"""
    # ✅ 【修改1】增加兼容参数：in_channels=通道数, shortcut, g，完全匹配C3k2的传参格式
    def __init__(self, in_channels, shortcut=False, g=1, reduction=16):
        super(RCSSC, self).__init__()
        n_feat = in_channels  # 核心：通道数直接等于输入通道数，和C3k2完全对齐
        pooling_r = 4
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),
        )
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(n_feat)
        )
        self.SA = spatial_attn_layer()
        self.CA = CALayer(n_feat, reduction)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU()
        self.tail = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1)
    def forward(self, x):
        res = x
        x = self.head(x)
        sa_branch = self.SA(x)
        ca_branch = self.CA(x)
        x1 = torch.cat([sa_branch, ca_branch], dim=1)
        x1 = self.conv1x1(x1)
        x2 = torch.sigmoid(torch.add(x, F.interpolate(self.SC(x), x.size()[2:])))
        out = torch.mul(x1, x2)
        out = self.tail(out)
        out = out + res
        out = self.ReLU(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 64, 64).to(device)
    rcssc = RCSSC(64).to(device) # 测试兼容传参
    y = rcssc(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)