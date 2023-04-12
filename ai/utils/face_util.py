import math
from typing import Tuple

import numpy as np
import cv2


class FaceUtil(object):
    """人脸图像处理工具"""

    @classmethod
    def crop(cls, img: np.ndarray, box: np.ndarray, landmark: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        裁剪人脸图像
        :param img: 要裁剪的人脸的图像
        :param box: 人脸框坐标
        :param landmark: 人脸关键点坐标
        :return: 裁剪后的人脸图像
        """
        # 解析人脸框坐标
        # (左上_x, 左上_y, 右下_x, 右下_y)
        box_l_x, box_l_y, box_r_x, box_r_y = tuple(box)
        # 裁剪图片至人脸框
        crop_img = np.array(img)[box_l_y:box_r_y, box_l_x:box_r_x]
        # 调整人脸关键点位置(相对于crop_img)
        landmark = np.reshape(landmark, (5, 2))  # 转换为5x2矩阵，每行是关键点的(x,y)坐标
        landmark = landmark - np.array([box_l_x, box_l_y])  # 调整坐标，以crop_img左上角为原点

        return crop_img, landmark

    @classmethod
    def alignment(cls, img: np.ndarray, landmark: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        人脸对齐
        :param img: 要对齐的人脸图像
        :param landmark: 人脸关键点坐标
        :return: 对齐后的(人脸图像，人脸关键点坐标)
        """
        # 计算左眼和右眼的坐标差值
        dx = landmark[0, 0] - landmark[1, 0]
        dy = landmark[0, 1] - landmark[1, 1]
        # 计算眼睛连线相对于水平线的倾斜角
        if dx == 0:
            angle = 0
        else:
            # 计算它的角度制
            angle = math.atan(dy / dx) * 180 / math.pi

        # 获取图片的宽度和高度
        img_w = img.shape[1]
        img_h = img.shape[0]

        # 获取图片的中心坐标
        center = (img_w // 2, img_h // 2)

        # 生成旋转变换矩阵
        RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)

        # 进行仿射变换，获得新图像
        new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

        # 将旋转变换矩阵转换为numpy形式
        RotationMatrix = np.array(RotationMatrix)

        # 构建变换后的人脸关键点坐标矩阵
        new_landmark = []
        for i in range(landmark.shape[0]):
            # 坐标数组
            pts = []
            # 对x坐标进行变换
            pts.append(
                RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
            # 对y坐标进行变换
            pts.append(
                RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])

            new_landmark.append(pts)

        new_landmark = np.array(new_landmark)

        return new_img, new_landmark
