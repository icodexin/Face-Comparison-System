from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image


class ImageUtil(object):
    """图像处理工具"""

    @classmethod
    def letterbox_image(cls, image: np.ndarray, size: Tuple[int, int]):
        """
        对输入图像进行不失真地调整大小
        :param image: 输入图像
        :param size: 调整后的大小
        :return: 调整后的图像
        """
        ih, iw, _ = np.shape(image)
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = cv2.resize(image, (nw, nh))
        new_image = np.ones([size[1], size[0], 3]) * 128
        new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
        return new_image

    @classmethod
    def resize_image(cls, image: np.ndarray, size: Tuple[int, int], letterbox_image: bool) -> np.ndarray:
        """
        调整图像大小
        :param image: 要调整的图像
        :param size: 调整后的大小
        :param letterbox_image: 是否进行不失真的调整
        :return: 调整后的图像
        """
        w, h = size
        if letterbox_image:
            new_image = ImageUtil.letterbox_image(image, size)
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image

    @classmethod
    def normalize(cls, image: np.ndarray):
        """
        图像归一化
        :param image: 输入图像
        :return: 归一后的图像
        """
        image /= 255.
        return image

    @classmethod
    def preprocess(cls, image):
        """
        预处理图像
        :param image: 待处理的图像
        :return: 处理后的图像
        """
        image -= np.array((104, 117, 123), np.float32)
        return image
