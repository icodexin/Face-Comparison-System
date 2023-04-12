from typing import Dict

import yaml
import numpy as np

from ai.face.detection import FaceDetector
from ai.face.feature import FaceFeatureExtractor
from ai.utils import print_config, FaceUtil


class FaceRecognizer(object):
    """人脸1:1识别器"""

    def __init__(self, config_path: str = './config.yaml'):
        """
        初始化人脸1:1识别器
        :param config_path: 配置文件路径
        """
        # 打开配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            print("\033[1;31m[Error]\033[0m Can not find or open config file.")

        # 获取配置信息
        use_cuda = config['use_cuda']  # 是否使用Cuda
        detection_config = config['detection_config']  # 获取 detection 配置信息
        feature_config = config['feature_config']  # 获取 feature 配置信息

        # 打印配置信息
        print(f'use_cuda: {use_cuda}')
        print_config('Face Detection Configs', detection_config)
        print_config('Feature Extractor Configs', feature_config)

        # 将配置信息传递给 detector 和 feature_extractor
        detection_config.update({'cuda': use_cuda})
        feature_config.update({'cuda': use_cuda})
        self.detector = FaceDetector(**detection_config)
        self.feature_extractor = FaceFeatureExtractor(**feature_config)

    def get_face_encoding(self, image):
        """
        获取人脸图片的编码
        :param image: 带有人脸的图片
        :return:
        """
        # 人脸检测
        boxes_conf_landms = self.detector.detect_image(image)

        # 人脸数检查
        if len(boxes_conf_landms) < 1:
            print('error_0')
            return {
                'success': False,  # 操作失败
                'error_code': 0,  # 没有人脸
            }
        elif len(boxes_conf_landms) > 1:
            print('error_1')
            return {
                'success': False,  # 操作失败
                'error_code': 1,  # 多余1个人脸
            }


        # 取出唯一的人脸信息，同时对数据取整
        face_info = np.array(boxes_conf_landms[0], dtype=int)

        # 裁剪人脸图像
        crop_img, landmark = FaceUtil.crop(image, face_info[:4], face_info[5:])
        # 对齐人脸图像
        crop_img, landmark = FaceUtil.alignment(crop_img, landmark)

        # 获取人脸编码
        encoding = self.feature_extractor.get_encoding(crop_img)

        return encoding

    def detect_image(self, image1, image2):
        """
        检测两张人脸图片的相似度
        :param image1: 第1张图片
        :param image2: 第2张图片
        :return: todo ?
        """
        encoding1 = self.get_face_encoding(image1)
        encoding2 = self.get_face_encoding(image2)

        return np.linalg.norm(encoding1 - encoding2)



if __name__ == "__main__":
    import cv2

    a = FaceRecognizer()
    image1 = cv2.imread('/Users/yangxin/Downloads/a.jpeg')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread('/Users/yangxin/Downloads/b.jpeg')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    aaa = a.detect_image(image1, image2)
    print(aaa)
