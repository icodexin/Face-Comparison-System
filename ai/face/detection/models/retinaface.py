import torch
import torch.nn as nn

from layers import FPN, SSH


class ClassHead(nn.Module):
    """分类预测"""

    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        """
        分类预测
        :param in_channels: 输入通道数
        :param num_anchors: 先验框数量
        """
        super(ClassHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        """
        前向推理
        :param x: 输入张量
        :return: 输出张量
        """
        out = self.conv1x1(x)  # 调整通道数
        out = out.permute(0, 2, 3, 1).contiguous()  # 将通道调整至最后一维
        return out.view(out.shape[0], -1, 2)  # [batch_size, num_anchors, is_contain_face?]


class BboxHead(nn.Module):
    """人脸框回归预测"""

    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        """
        人脸框回归预测
        :param in_channels: 输入通道数
        :param num_anchors: 先验框数量
        """
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        """
        前向推理
        :param x: 输入张量
        :return: 输出张量
        """
        out = self.conv1x1(x)  # 调整通道数
        out = out.permute(0, 2, 3, 1).contiguous()  # 将通道调整至最后一维
        return out.view(out.shape[0], -1, 4)  # [batch_size, num_anchors, [框中心_x, 框中心_y, 框_h, 框_w]]


class LandmarkHead(nn.Module):
    """人脸关键点回归预测"""

    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        """
        人脸关键点回归预测
        :param in_channels: 输入通道数
        :param num_anchors: 先验框数量
        """
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        """
        前向推理
        :param x: 输入张量
        :return: 输出张量
        """
        out = self.conv1x1(x)  # 调整通道数
        out = out.permute(0, 2, 3, 1).contiguous()  # 将通道调整至最后一维
        return out.view(out.shape[0], -1, 10)  # [batch_size, num_anchors, 5个关键点的坐标]

