import math

import torch
from torch import nn


class Conv2d_cd(nn.Module):
    """Central-difference convolution represented by an equivalent 3x3 kernel."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def get_weight(self):
        weight = self.conv.weight
        flat_weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        diff_weight = flat_weight.clone()
        diff_weight[:, :, 4] = flat_weight[:, :, 4] - flat_weight.sum(2)
        return diff_weight.reshape_as(weight), self.conv.bias


class Conv2d_ad(nn.Module):
    """Angular-difference convolution represented by an equivalent 3x3 kernel."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def get_weight(self):
        weight = self.conv.weight
        flat_weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        permutation = [3, 0, 1, 6, 4, 2, 7, 8, 5]
        diff_weight = flat_weight - self.theta * flat_weight[:, :, permutation]
        return diff_weight.reshape_as(weight), self.conv.bias


class Conv2d_rd(nn.Module):
    """Radial-difference convolution retained for compatibility with older imports."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def forward(self, x):
        if math.fabs(self.theta) < 1e-8:
            return self.conv(x)

        weight = self.conv.weight
        flat_weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        diff_weight = torch.zeros(
            weight.shape[0], weight.shape[1], 25, device=weight.device, dtype=weight.dtype
        )
        diff_weight[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = flat_weight[:, :, 1:]
        diff_weight[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -flat_weight[:, :, 1:] * self.theta
        diff_weight[:, :, 12] = flat_weight[:, :, 0] * (1 - self.theta)
        return nn.functional.conv2d(
            x,
            diff_weight.view(weight.shape[0], weight.shape[1], 5, 5),
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
        )


class Conv2d_hd(nn.Module):
    """Horizontal-difference convolution represented by an equivalent 3x3 kernel."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def get_weight(self):
        weight = self.conv.weight
        flat_weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        diff_weight = torch.zeros_like(flat_weight)
        diff_weight[:, :, [0, 3, 6]] = flat_weight[:, :, :3]
        diff_weight[:, :, [2, 5, 8]] = -flat_weight[:, :, :3]
        return diff_weight.reshape_as(weight), self.conv.bias


class Conv2d_vd(nn.Module):
    """Vertical-difference convolution represented by an equivalent 3x3 kernel."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def get_weight(self):
        weight = self.conv.weight
        flat_weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        diff_weight = torch.zeros_like(flat_weight)
        diff_weight[:, :, [0, 1, 2]] = flat_weight[:, :, :3]
        diff_weight[:, :, [6, 7, 8]] = -flat_weight[:, :, :3]
        return diff_weight.reshape_as(weight), self.conv.bias


class DEConv(nn.Module):
    """Detail-enhanced convolution with an optional deploy-time fused kernel."""

    def __init__(self, in_channels, out_channels=None, bias=True, deploy=False):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        else:
            self.conv1_1 = Conv2d_cd(in_channels, out_channels, 3, bias=bias)
            self.conv1_2 = Conv2d_hd(in_channels, out_channels, 3, bias=bias)
            self.conv1_3 = Conv2d_vd(in_channels, out_channels, 3, bias=bias)
            self.conv1_4 = Conv2d_ad(in_channels, out_channels, 3, bias=bias)
            self.conv1_5 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)

    def get_equivalent_kernel_bias(self):
        """Return the single 3x3 kernel and bias equivalent to all five branches."""
        if self.deploy:
            return self.reparam_conv.weight, self.reparam_conv.bias

        branches = [
            self.conv1_1.get_weight(),
            self.conv1_2.get_weight(),
            self.conv1_3.get_weight(),
            self.conv1_4.get_weight(),
            (self.conv1_5.weight, self.conv1_5.bias),
        ]
        kernel = sum(weight for weight, _ in branches)
        bias = torch.zeros(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
        for _, branch_bias in branches:
            if branch_bias is not None:
                bias = bias + branch_bias
        return kernel, bias

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"DEConv expected {self.in_channels} input channels, got {x.shape[1]}. "
                "Fix the model YAML channel definition instead of adapting channels in forward()."
            )
        if self.deploy:
            return self.reparam_conv(x)

        kernel, bias = self.get_equivalent_kernel_bias()
        kernel = kernel.to(device=x.device, dtype=x.dtype)
        bias = bias.to(device=x.device, dtype=x.dtype)
        return nn.functional.conv2d(x, kernel, bias, stride=1, padding=1)

    def switch_to_deploy(self):
        """Replace the five training branches with one equivalent Conv2d."""
        if self.deploy:
            return self

        kernel, bias = self.get_equivalent_kernel_bias()
        reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1, bias=True).to(
            device=kernel.device, dtype=kernel.dtype
        )
        with torch.no_grad():
            reparam_conv.weight.copy_(kernel)
            reparam_conv.bias.copy_(bias)

        self.reparam_conv = reparam_conv
        del self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4, self.conv1_5
        self.deploy = True
        return self

    def fuse(self):
        """Fuse the training branches for export or deployment."""
        return self.switch_to_deploy()


class LiteDEConv(nn.Module):
    """Lightweight detail-enhanced convolution for resource-constrained training.

    The full DEConv uses five dense 3x3 branches. This variant keeps one dense
    3x3 branch and adds a depthwise-separable detail branch, reducing parameters
    while preserving a learnable local-detail path.
    """

    def __init__(self, in_channels, out_channels=None, bias=True, detail_init=0.1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)
        self.detail_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.detail_pw = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_init)))

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"LiteDEConv expected {self.in_channels} input channels, got {x.shape[1]}. "
                "Fix the model YAML channel definition instead of adapting channels in forward()."
            )
        return self.base(x) + self.detail_scale * self.detail_pw(self.detail_dw(x))
