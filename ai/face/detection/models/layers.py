from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_channels: int, out_channels: int, stride: int = 1, leaky: float = 0.):
    """
    带有 BN 和 ReLU 的标准卷积
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param stride: 步长
    :param leaky: LeakyReLU 中的 leaky 因子，用于控制负区间的倾斜角度
    :return: 卷积后的特征图
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(in_channels: int, out_channels: int, stride: int):
    """
    带有 BN 的标准卷积
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param stride: 步长
    :return: 卷积后的特征图
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


def conv_bn1X1(in_channels: int, out_channels: int, stride: int, leaky: float = 0.):
    """
    1X1逐点卷积
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param stride: 步长
    :param leaky: LeakyReLU 中的 leaky 因子，用于控制负区间的倾斜角度
    :return: 卷积后的特征图
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class FPN(nn.Module):
    """
    Feature Pyramid Network

    特征金字塔
    """

    def __init__(self, in_channels_list: List[int], out_channels: int):
        """
        Feature Pyramid Network
        :param in_channels_list: 输入通道数列表
        :param out_channels: 输出通道数
        """
        super(FPN, self).__init__()
        leaky = 0.
        if out_channels <= 64:
            leaky = 0.1

        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, x: Dict[str, torch.Tensor]):
        """
        前向推理
        :param x: 输入张量字典
        :return: 输出张量
        """
        y = list(x.values())

        output1 = self.output1(y[0])  # C3通道数调整
        output2 = self.output2(y[1])  # C4通道数调整
        output3 = self.output3(y[2])  # C5通道数调整，获得P5

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode='nearest')  # 上采样
        output2 = output2 + up3
        output2 = self.merge2(output2)  # 特征融合，获得P4

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode='nearest')  # 上采样
        output1 = output1 + up2
        output1 = self.merge1(output1)  # 特征融合，获得P3

        return [output1, output2, output3]  # 传回[P3, P4, P5]


class SSH(nn.Module):
    """Single Stage Headless Face Detector"""

    def __init__(self, in_channels: int, out_channels: int):
        """
        Single Stage Headless Face Detector
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super(SSH, self).__init__()
        if out_channels % 4 != 0:
            raise ValueError(f"Expect out channel % 4 == 0, but we got {out_channels % 4}")

        leaky = 0.
        if out_channels <= 64:
            leaky = 0.1

        self.conv3X3 = conv_bn_no_relu(in_channels, out_channels // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channels, out_channels // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channels // 4, out_channels // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channels // 4, out_channels // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channels // 4, out_channels // 4, stride=1)

    def forward(self, x: torch.Tensor):
        """
        前向推理
        :param x: 输入张量
        :return: 输出张量
        """
        conv3X3 = self.conv3X3(x)

        # 使用2个3x3卷积代替5x5卷积
        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        # 使用3个3x3卷积代替7x7卷积
        # 注意，第一次3x3卷积共用5x5中的第一次3x3卷积
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)

        return F.relu(out)
