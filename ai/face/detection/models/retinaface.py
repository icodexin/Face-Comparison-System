from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import _utils

from .layers import FPN, SSH
from ai.backbone import MobileNet025


class ClassHead(nn.Module):
    """分类预测"""

    def __init__(self, in_channels: int = 512, num_anchors: int = 2):
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

    def __init__(self, in_channels: int = 512, num_anchors: int = 2):
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

    def __init__(self, in_channels: int = 512, num_anchors: int = 2):
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


class RetinaFace(nn.Module):
    """RetinaFace网络"""

    def __init__(self, cfg: Dict[str, Any], pretrained=False, mode='train'):
        """
        初始化RetinaFace网络
        :param cfg: 配置信息字典
        :param pretrained: 网络是否有预训练权重
        :param mode: 模式
        """
        super(RetinaFace, self).__init__()
        backbone = None
        # 选择mobilenet0.25,resnet50作为主干网络
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNet025()
            if pretrained:
                checkpoint = torch.load("./model_data/mobilenetV1x0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in checkpoint['state_dict'].items():
                    name = key[7:]
                    new_state_dict[name] = value
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=pretrained)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        # 获得每个初步有效特征层的通道数
        in_channels_list = [cfg['in_channel'] * 2, cfg['in_channel'] * 4, cfg['in_channel'] * 8]
        # 利用初步有效特征层构建特征金字塔FPN
        self.fpn = FPN(in_channels_list, cfg['out_channel'])
        # 利用SSH模块提高模型感受野
        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])

        self.ClassHead = self._make_class_head(fpn_num=3, in_channels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, in_channels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, in_channels=cfg['out_channel'])

        self.mode = mode

    @staticmethod
    def _make_class_head(fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2):
        """"构建ClassHead分类回归器"""
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(ClassHead(in_channels, anchor_num))
        return classhead

    @staticmethod
    def _make_bbox_head(fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2):
        """"构建BboxHead人脸框回归器"""
        bboxhead = nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(BboxHead(in_channels, anchor_num))
        return bboxhead

    @staticmethod
    def _make_landmark_head(fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2):
        """"构建LandmarkHead人脸关键点回归器"""
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(in_channels, anchor_num))
        return landmarkhead

    def forward(self, inputs: torch.Tensor):
        """
        前向推理
        :param inputs: 输入张量
        :return: 输出张量
        """
        # 通过主干网络获得3个初步特征层
        # 层级 HxWxC
        # C3 80x80x64
        # C4 40x40x128
        # C5 20x20x256
        out = self.body.forward(inputs)

        # 通过特征金字塔FPN获得3个有效特征层
        # 层级 HxWxC
        # P3 80x80x64
        # P4 40x40x64
        # P5 20x20x64
        fpn = self.fpn.forward(out)

        # 通过SSH模块加强感受野
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        # 将所有结果进行堆叠
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.mode == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        return output