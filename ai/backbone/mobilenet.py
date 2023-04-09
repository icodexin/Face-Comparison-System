import torch
import torch.nn as nn
import torchinfo


class DepthSepConv2d(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, leaky: float = 0.1):
        """
        深度可分离卷积
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 深度卷积的步长
        :param leaky: LeakyReLU 中的 leaky 因子，用于控制负区间的倾斜角度
        """
        super(DepthSepConv2d, self).__init__()
        # 深度卷积
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=leaky, inplace=True),
        )
        # 逐点卷积
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaky, inplace=True),
        )

    def forward(self, x: torch.Tensor):
        """
        前向推理
        :param x: 输入张量
        :return: 推理后的张量
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def conv_bn(in_channels: int, out_channels: int, stride: int, leaky: float = 0.):
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


class MobileNetV1(nn.Module):
    def __init__(self, n_class=1000, alpha=1.):
        """
        初始化 MobileNetV1
        :param n_class: 分类数
        :param alpha: 宽度超参数
        """
        super(MobileNetV1, self).__init__()
        self.n_class = n_class
        # 一个标准卷积+5个深度可分离卷积
        self.stage1 = nn.Sequential(
            conv_bn(3, int(alpha * 32), 2),
            DepthSepConv2d(int(alpha * 32), int(alpha * 64), 1),
            DepthSepConv2d(int(alpha * 64), int(alpha * 128), 2),
            DepthSepConv2d(int(alpha * 128), int(alpha * 128), 1),
            DepthSepConv2d(int(alpha * 128), int(alpha * 256), 2),
            DepthSepConv2d(int(alpha * 256), int(alpha * 256), 1),
        )  # In here, you can get C3 for FPN
        # 6个深度可分离卷积
        self.stage2 = nn.Sequential(
            DepthSepConv2d(int(alpha * 256), int(alpha * 512), 2),
            DepthSepConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSepConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSepConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSepConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSepConv2d(int(alpha * 512), int(alpha * 512), 1),
        )  # In here, you can get C4 for FPN
        # 2个深度可分离卷积
        self.stage3 = nn.Sequential(
            DepthSepConv2d(int(alpha * 512), int(alpha * 1024), 2),
            DepthSepConv2d(int(alpha * 1024), int(alpha * 1024), 1),  # todo have problem
        )  # In here, you can get C5 for FPN
        # 平均池化
        self.avg = nn.AdaptiveAvgPool2d(1)
        # 全连接
        self.fc = nn.Linear(int(alpha * 1024), n_class)

    def forward(self, x: torch.Tensor):
        """
        前向推理
        :param x: 输入张量
        :return: 输出张量
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = MobileNetV1(alpha=0.25)
    torchinfo.summary(model, input_size=(1, 3, 640, 640))
