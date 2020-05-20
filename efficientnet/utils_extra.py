# Author: Zylo117

import math

from torch import nn
import torch.nn.functional as F

"""
假设是128x128的feature map，用vaild padding，舍弃边缘和角落，那么影响的数据点为128*4-4，508个，对原map的信息丢失率为508/（128*128）≈ 3.1%。
同样的方法，当feature map只有6x6的时候，信息丢失率则是55.6%。
所以我们可以看出来，same padding在小feature map上是必要的，否则将会丢失将近过半的信息！

接下来就对卷积和池化分别做了SamePadding的实现。
"""

class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    卷积的时候进行padding部分的操作。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        h_step = math.ceil(w / self.stride[1])
        v_step = math.ceil(h / self.stride[0])
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

        extra_h = h_cover_len - w
        extra_v = v_cover_len - h

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    池化的时候进行padding部分的操作
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        # 为什么要扩充成2个 int => [int,int] 和 [int] => [int,int]
        # 是因为，传入的值可能会是stride=[1,3],第0维1步滑动，第1维3步滑动，
        # 同理，kernel_size也有[1,3]这样窗口大小的核。
        # 所以padding的时候，要根据两个维度的大小分别计算需要padding的数值。
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        h_step = math.ceil(w / self.stride[1])
        v_step = math.ceil(h / self.stride[0])
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

        extra_h = h_cover_len - w
        extra_v = v_cover_len - h

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])
        # [left, right, top, bottom]：传入四元素tuple(pad_l, pad_r, pad_t, pad_b)，
		# 指的是（左填充，右填充，上填充，下填充），其数值代表填充次数

        x = self.pool(x)
        return x
