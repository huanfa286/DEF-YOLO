import torch
import torch.nn as nn


class AttentionWeight(nn.Module):
    """注意力权重计算模块，用于生成空间维度上的注意力权重
    功能：通过对输入特征的最大值和平均值进行聚合，经卷积和激活后生成权重，实现空间注意力机制
    """

    def __init__(self, channel, kernel_size=7):
        super(AttentionWeight, self).__init__()
        padding = (kernel_size - 1) // 2  # 保持卷积前后特征尺寸不变
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)  # 融合最大池化和平均池化特征
        # 深度可分离卷积，减少参数同时捕捉局部空间信息
        self.conv2 = nn.Conv1d(channel, channel, kernel_size, padding=padding, groups=channel, bias=False)
        self.bn = nn.BatchNorm1d(channel)  # 批归一化，稳定训练
        self.sigmoid = nn.Sigmoid()  # 输出权重在0-1之间

    def forward(self, x):
        b, w, c, h = x.size()  # 获取输入特征尺寸：(batch, 宽度, 通道, 高度)
        # 沿宽度维度聚合特征（最大池化和平均池化），并拼接
        x_weight = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_weight = self.conv1(x_weight).view(b, c, h)  # 降维并调整形状
        # 经卷积、批归一化和激活函数生成权重
        x_weight = self.sigmoid(self.bn(self.conv2(x_weight)))
        x_weight = x_weight.view(b, 1, c, h)  # 调整权重形状以匹配输入
        return x * x_weight  # 应用注意力权重


class IIA(nn.Module):
    """交互式注意力模块（Interactive Attention Module）
    功能：通过在高度和宽度两个空间维度分别施加注意力，增强特征的空间相关性
    """
    # ======================================
    # ✅ 【唯一修改点1】：兼容C3k2的传参格式，核心逻辑完全不变
    # 原参数：def __init__(self, channel):
    # 新参数：def __init__(self, in_channels, shortcut=False, g=1):  完美匹配你的yaml传参
    # ======================================
    def __init__(self, in_channels, shortcut=False, g=1):
        super(IIA, self).__init__()
        self.attention = AttentionWeight(in_channels)  # 把传参赋值给原模块，仅此一行改动

    def forward(self, x):
        # 处理高度维度：调整维度顺序后应用注意力，再恢复原顺序
        x_h = x.permute(0, 3, 1, 2).contiguous()
        x_h = self.attention(x_h).permute(0, 2, 3, 1)
        # 处理宽度维度：调整维度顺序后应用注意力，再恢复原顺序
        x_w = x.permute(0, 2, 1, 3).contiguous()
        x_w = self.attention(x_w).permute(0, 2, 1, 3)

        # 将原始特征与两个维度的注意力特征相加，增强有效特征
        return x + x_h + x_w


class ChannelAttention(nn.Module):
    """通道注意力模块
    功能：通过聚合空间维度的信息（平均池化和最大池化），生成通道维度的注意力权重，增强重要通道特征
    """

    def __init__(self, inp, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 空间维度平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 空间维度最大池化
        # 压缩-激励结构：通过1x1卷积实现通道维度压缩和恢复
        self.fc1 = nn.Conv2d(inp, inp // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inp // ratio, inp, 1, bias=False)
        self.sigmoid = nn.Sigmoid()  # 输出通道权重在0-1之间

    def forward(self, x):
        # 计算平均池化路径的注意力
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 计算最大池化路径的注意力
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # 融合两种池化的注意力
        return self.sigmoid(out)  # 输出通道权重


device = torch.device('cuda:0')  # 定义计算设备为第一块GPU
if __name__ == '__main__':
    # 模块测试：实例化IIA模块并验证输入输出尺寸
    net = IIA(512)  # 测试兼容传参
    net = net.cuda()
    x = torch.rand(8, 512, 16, 16).to(device)  # 输入特征：(batch=8, channel=512, height=16, width=16)
    y = net(x)
    print(y.size())  # 输出应与输入尺寸一致：torch.Size([8, 512, 16, 16])