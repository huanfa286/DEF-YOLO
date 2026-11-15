import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from ultralytics.nn.modules import Conv

# 补全缺失的BaseModule父类
class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.deploy = False

class Block1x1(BaseModule):
    """GCBlock的1×1-1×1分支模块（训练双1×1卷积+BN，推理融合为单1×1卷积）"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 0,
                 deploy: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=True)
        else:
            # ✅ 修复点1：删除多余的 bias=bias 传参，Ultralytics Conv自动处理
            self.conv1 = Conv(in_channels, out_channels, 1, stride, padding, act=False)
            self.conv2 = Conv(out_channels, out_channels, 1, 1, padding, act=False)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    def _fuse_bn_tensor(self, conv: Conv):
        kernel = conv.conv.weight
        bias = conv.conv.bias if conv.conv.bias is not None else 0.
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = beta + (bias - running_mean) * gamma / std if conv.conv.bias is not None else beta - running_mean * gamma / std
        return fused_kernel, fused_bias

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        fused_kernel = torch.einsum('oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        fused_bias = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=self.padding, bias=True)
        self.conv.weight.data = fused_kernel
        self.conv.bias.data = fused_bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True

class Block3x3(BaseModule):
    """GCBlock的3×3-1×1分支模块（训练3×3+1×1+BN，推理融合为单3×3卷积）"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 0,
                 deploy: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=True)
        else:
            # ✅ 修复点2：删除多余的 bias=bias 传参，核心报错位置！
            self.conv1 = Conv(in_channels, out_channels, 3, stride, padding, act=False)
            self.conv2 = Conv(out_channels, out_channels, 1, 1, 0, act=False)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    def _fuse_bn_tensor(self, conv: Conv):
        return Block1x1._fuse_bn_tensor(self, conv)

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        fused_kernel = torch.einsum('oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        fused_bias = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=self.padding, bias=True)
        self.conv.weight.data = fused_kernel
        self.conv.bias.data = fused_bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True

class GCBlock(nn.Module):
    """✅ 最终完美可用版 GCBlock 核心模块（无报错+重参数化+多分支融合）"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 padding_mode: str = 'zeros',
                 act: bool = True,
                 deploy: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        assert kernel_size == 3, "GCBlock only supports kernel_size=3"
        assert padding == 1, "GCBlock only supports padding=1 for same output size"
        padding_11 = padding - kernel_size // 2
        self.relu = nn.ReLU(inplace=True) if act else nn.Identity()

        if deploy:
            self.reparam_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        else:
            self.path_residual = nn.BatchNorm2d(in_channels) if (out_channels == in_channels and stride == 1) else None
            # ✅ 修复点3：删除Block3x3/Block1x1里的bias=False传参
            self.path_3x3_1 = Block3x3(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding)
            self.path_3x3_2 = Block3x3(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding)
            self.path_1x1 = Block1x1(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding_11)

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, 'reparam_3x3'):
            return self.relu(self.reparam_3x3(inputs))
        id_out = self.path_residual(inputs) if self.path_residual is not None else 0.
        total_out = self.path_3x3_1(inputs) + self.path_3x3_2(inputs) + self.path_1x1(inputs) + id_out
        return self.relu(total_out)

    def get_equivalent_kernel_bias(self):
        self.path_3x3_1.switch_to_deploy()
        kernel3x3_1, bias3x3_1 = self.path_3x3_1.conv.weight.data, self.path_3x3_1.conv.bias.data
        self.path_3x3_2.switch_to_deploy()
        kernel3x3_2, bias3x3_2 = self.path_3x3_2.conv.weight.data, self.path_3x3_2.conv.bias.data
        self.path_1x1.switch_to_deploy()
        kernel1x1, bias1x1 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data
        kernel1x1_padded = self._pad_1x1_to_3x3_tensor(kernel1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.path_residual)
        equivalent_kernel = kernel3x3_1 + kernel3x3_2 + kernel1x1_padded + kernelid
        equivalent_bias = bias3x3_1 + bias3x3_2 + bias1x1 + biasid
        return equivalent_kernel, equivalent_bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0.
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, conv):
        if conv is None:
            return 0., 0.
        if isinstance(conv, Conv):
            kernel = conv.conv.weight
            running_mean = conv.bn.running_mean
            running_var = conv.bn.running_var
            gamma = conv.bn.weight
            beta = conv.bn.bias
            eps = conv.bn.eps
        else:
            assert isinstance(conv, (nn.SyncBatchNorm, nn.BatchNorm2d)), "Only BN layers are supported"
            if not hasattr(self, 'id_tensor'):
                kernel_value = np.zeros((self.out_channels, self.in_channels, 3, 3), dtype=np.float32)
                for i in range(self.out_channels):
                    kernel_value[i, i % self.in_channels, 1, 1] = 1.
                self.id_tensor = torch.from_numpy(kernel_value).to(conv.weight.device)
            kernel = self.id_tensor
            running_mean = conv.running_mean
            running_var = conv.running_var
            gamma = conv.weight
            beta = conv.bias
            eps = conv.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        return fused_kernel, fused_bias

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_3x3'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_3x3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('path_3x3_1')
        self.__delattr__('path_3x3_2')
        self.__delattr__('path_1x1')
        if hasattr(self, 'path_residual'):
            self.__delattr__('path_residual')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

# 测试用例（可直接运行✅ 无报错）
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)
    model = GCBlock(64, 128).to(device)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)