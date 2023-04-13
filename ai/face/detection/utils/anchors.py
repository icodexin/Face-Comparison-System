from itertools import product
from math import ceil
from typing import Tuple, Dict, Any, List

import numpy as np
import torch


def decode(loc, priors, variances):
    """
    中心解码，宽高解码
    :param loc:
    :param priors:
    :param variances:
    :return:
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """
    关键点解码
    :param pre:
    :param priors:
    :param variances:
    :return:
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


class Anchors(object):
    """先验框"""

    def __init__(self, config: Dict[str, Any], image_size: Tuple[int, int]):
        """
        先验框
        :param config: 配置信息字典
        :param image_size: 图像大小
        """
        # P3,P4,P5的2个先验框最小边长
        self.min_sizes: List[List[int]] = config['min_sizes']
        # P3,P4,P5特征图的长宽压缩倍数
        self.steps: List[int] = config['steps']
        # todo 是否clip
        self.clip: bool = config['clip']
        # 特征图的原始尺寸
        self.image_size: Tuple[int, int] = image_size

        # P3,P4,P5有效特征层的高和宽
        self.feature_maps: List[List[int]] = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def get_anchors(self):
        """获得先验框列表"""
        anchor_list = []
        # 依次处理P3,P4,P5
        for index, feature_map in enumerate(self.feature_maps):
            min_size = self.min_sizes[index]
            # 对特征层的每个网格进行处理
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                # 每个网格有两个先验框
                for box_size in min_size:
                    s_kx = box_size / self.image_size[1]
                    s_ky = box_size / self.image_size[0]
                    dense_cx = [x * self.steps[index] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[index] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchor_list += [cx, cy, s_kx, s_ky]  # [中心_x, 中心_y, 宽, 高]

        output = torch.Tensor(anchor_list).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


# if __name__ == "__main__":
#     from config import cfg_mnet
#     import matplotlib
#     import matplotlib.pyplot as plt
#
#     # 设定图片大小
#     cfg_mnet['image_size'] = 640
#     cfg = cfg_mnet
#
#     # 获得原始先验框
#     raw_anchors = Anchors(cfg, image_size=(cfg['image_size'], cfg['image_size'])).get_anchors()
#     # 将原始先验框坐标转换为[左上x, 左上y, 右下x, 右下y]
#     anchors = np.zeros_like(raw_anchors[:, :4])
#     anchors[:, 0] = raw_anchors[:, 0] - raw_anchors[:, 2] / 2
#     anchors[:, 1] = raw_anchors[:, 1] - raw_anchors[:, 3] / 2
#     anchors[:, 2] = raw_anchors[:, 0] + raw_anchors[:, 2] / 2
#     anchors[:, 3] = raw_anchors[:, 1] + raw_anchors[:, 3] / 2
#
#     anchors = anchors[-800:] * cfg['image_size']
#
#     # 先验框中心绘制
#     center_x = (anchors[:, 0] + anchors[:, 2]) / 2
#     center_y = (anchors[:, 1] + anchors[:, 3]) / 2
#     fig = plt.figure()
#     ax = fig.add_subplot(121)
#     plt.ylim(-300, 900)
#     plt.xlim(-300, 900)
#     ax.invert_yaxis()
#     plt.scatter(center_x, center_y)
#     # 先验框宽高绘制
#     box_widths = anchors[0:2, 2] - anchors[0:2, 0]
#     box_heights = anchors[0:2, 3] - anchors[0:2, 1]
#     for i in [0, 1]:
#         rect = plt.Rectangle((anchors[i, 0], anchors[i, 1]), box_widths[i], box_heights[i], color="r", fill=False)
#         ax.add_patch(rect)
#     # 先验框中心绘制
#     ax = fig.add_subplot(122)
#     plt.ylim(-300, 900)
#     plt.xlim(-300, 900)
#     ax.invert_yaxis()  # y轴反向
#     plt.scatter(center_x, center_y)
#
#     # 对先验框调整获得预测框
#     mbox_loc = np.random.randn(800, 4)
#     mbox_ldm = np.random.randn(800, 10)
#
#     anchors[:, :2] = (anchors[:, :2] + anchors[:, 2:]) / 2
#     anchors[:, 2:] = (anchors[:, 2:] - anchors[:, :2]) * 2
#
#     mbox_loc = torch.Tensor(mbox_loc)
#     anchors = torch.Tensor(anchors)
#     cfg_mnet['variance'] = torch.Tensor(cfg_mnet['variance'])
#     decode_bbox = decode(mbox_loc, anchors, cfg_mnet['variance'])
#
#     box_widths = decode_bbox[0: 2, 2] - decode_bbox[0: 2, 0]
#     box_heights = decode_bbox[0: 2, 3] - decode_bbox[0: 2, 1]
#
#     for i in [0, 1]:
#         rect = plt.Rectangle((decode_bbox[i, 0], decode_bbox[i, 1]), box_widths[i], box_heights[i], color="r",
#                              fill=False)
#         plt.scatter((decode_bbox[i, 2] + decode_bbox[i, 0]) / 2, (decode_bbox[i, 3] + decode_bbox[i, 1]) / 2, color="b")
#         ax.add_patch(rect)
#
#     plt.show()
