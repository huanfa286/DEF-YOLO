import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class AddShift_mp_module(nn.Module):
    def __init__(self, big_kernel, small_kernel, repN, out_n, N_rep):
        super().__init__()
        self.big_kernel = big_kernel
        self.small_kernel = small_kernel
        self.repN = repN
        self.out_n = out_n
        self.N_rep = N_rep
        self.nk = int(big_kernel / small_kernel) if big_kernel % small_kernel ==0 else big_kernel//small_kernel +1

    def forward(self, x, b, h, w):
        x1 = x[:, 0:self.repN, :, :].contiguous()
        x2 = x[:, self.repN:2*self.repN, :, :].contiguous()
        x3 = x[:, 2*self.repN:3*self.repN, :, :].contiguous()
        return x1, x2, x3

use_sync_bn = True
def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

class ShifthWiseConv2dImplicit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 big_kernel=51,
                 small_kernel=3,
                 stride=1,
                 group=1,
                 bn=True,
                 use_small_conv=True,
                 ghost_ratio=0.23,
                 N_path=2,
                 N_rep=4,
                 bias=False):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
        self.in_channels = in_channels

        self.ghost_ratio = ghost_ratio
        self.ghostN = int(np.round(in_channels * ghost_ratio))
        self.ghostN = max(1, min(self.ghostN, in_channels-1))
        self.repN = in_channels - self.ghostN
        assert self.repN + self.ghostN == in_channels, f'通道总数不匹配'

        np.random.seed(123)
        ghost = np.random.choice(in_channels, self.ghostN, replace=False).tolist()
        ghost.sort()
        rep = list(set(range(in_channels)) - set(ghost))
        rep.sort()
        assert len(rep) == self.repN and len(ghost) == self.ghostN, '通道索引错误'
        
        # ========== 修复1：将索引张量转为模型缓冲区，自动跟随模型迁移cuda/cpu，根治设备错位 ==========
        self.register_buffer('ghost', torch.IntTensor(ghost))
        self.register_buffer('rep', torch.IntTensor(rep))

        out_n = self.repN * self.nk
        self.LoRA = None
        self.LoRAs = nn.ModuleList([
            nn.Conv2d(self.repN, out_n,
                      kernel_size=small_kernel, stride=stride,
                      padding=small_kernel//2, groups=self.repN,
                      bias=False)
            for _ in range(N_path)
        ])

        self.device = None
        print(f'ghost_ratio={ghost_ratio}, repN={self.repN}, ghostN={self.ghostN}, N_rep={N_rep}, N_path={N_path}')

        self.use_bn = bn
        self.loras = AddShift_mp_module(big_kernel, small_kernel, self.repN, out_n, N_rep)
        if bn:
            self.bn_lora1 = get_bn(self.repN)
            self.bn_lora2 = get_bn(self.repN)
            self.bn_small = get_bn(self.repN)
        else:
            self.bn_lora1 = None
            self.bn_lora2 = None
            self.bn_small = None

        self.inputsize = True

    def ghost_mask(self):
        weight = 0
        for name, tensor in self.named_parameters():
            if len(tensor.size()) == 4:
                weight += torch.sum(torch.abs(tensor.detach()), (1, 2, 3))
        weight = torch.sum((weight > 0).reshape(-1, self.nk), 1)
        ghost = (weight == 0).reshape(1, -1, 1, 1).float()
        return ghost

    def forward(self, inputs):
        if self.inputsize:
            print(f'shape:{inputs.shape}; kernel:{self.kernels}; in_ch:{self.in_channels}')
            self.inputsize = False

        ori_b, ori_c, ori_h, ori_w = inputs.shape
        assert ori_c == self.in_channels, f'输入通道数不匹配'

        # ========== 修复2：强制索引张量与输入张量同设备，双重保险 永不报错 ==========
        self.ghost = self.ghost.to(inputs.device)
        self.rep = self.rep.to(inputs.device)

        ghost_inputs = torch.index_select(inputs, 1, self.ghost)
        rep_inputs = torch.index_select(inputs, 1, self.rep)

        lora1_x = 0
        lora2_x = 0
        small_x = 0
        out = 0
        if self.LoRA is None:
            for split_conv in self.LoRAs:
                xx = split_conv(rep_inputs)
                out += xx
        else:
            out = self.LoRA(rep_inputs)

        x1, x2, x3 = self.loras(out, ori_b, ori_h, ori_w)
        lora1_x += x1
        lora2_x += x2
        small_x += x3

        if self.use_bn:
            lora1_x = self.bn_lora1(lora1_x)
            lora2_x = self.bn_lora2(lora2_x)
            small_x = self.bn_small(small_x)

        lora1_x = lora1_x[:, :, :ori_h, :ori_w].contiguous()
        lora2_x = lora2_x[:, :, :ori_h, :ori_w].contiguous()
        small_x = small_x[:, :, :ori_h, :ori_w].contiguous()

        x = lora1_x + lora2_x + small_x + rep_inputs
        x = torch.cat([x, ghost_inputs], dim=1)
        
        assert x.shape == inputs.shape, f'输入输出尺寸不匹配: {inputs.shape} → {x.shape}'
        return x

    def shift(self, kernels):
        mink, maxk = min(kernels), max(kernels)
        nk = math.ceil(maxk / mink)
        padding = mink - 1
        mid = maxk // 2
        real_pad = []
        for i in range(nk):
            extra_pad = mid - i * mink - padding
            real_pad.append(extra_pad)
        return padding, real_pad

    def merge_branches(self):
        if self.LoRA is None:
            bias = True if self.LoRAs[0].bias else False
            LoRA = nn.Conv2d(in_channels=self.LoRAs[0].in_channels,
                             out_channels=self.LoRAs[0].out_channels,
                             kernel_size=self.LoRAs[0].kernel_size,
                             stride=self.LoRAs[0].stride,
                             padding=self.LoRAs[0].padding,
                             dilation=self.LoRAs[0].dilation,
                             groups=self.LoRAs[0].groups,
                             bias=bias)
            weight, biasdata = 0, 0
            for merged_conv in self.LoRAs:
                weight += merged_conv.weight.data
                if bias:
                    biasdata += merged_conv.bias.data
            LoRA.weight.data = weight
            if bias:
                LoRA.bias.data = biasdata
            self.LoRA = LoRA
            self.__delattr__('LoRAs')

class SWConv(nn.Module):
    def __init__(self, c, k=51, s=1):
        super().__init__()
        self.swconv = ShifthWiseConv2dImplicit(in_channels=c, out_channels=c, big_kernel=k, stride=s)

    def forward(self, x):
        return self.swconv(x)