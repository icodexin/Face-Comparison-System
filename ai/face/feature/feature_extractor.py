import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ai.face.feature.models import FaceNet
from ai.utils import ImageUtil


class FaceFeatureExtractor(object):
    """人脸特征提取器"""
    _defaults = {
        # facenet 训练完的权值路径
        "model_path": "../model_data/facenet_mobilenet.pth",
        # 输入图片的大小。
        "input_shape": [160, 160, 3],
        # 所使用到的主干特征提取网络
        "backbone": "mobilenet",
        # 是否进行不失真的resize
        "letterbox_image": True,
        # 是否使用Cuda
        "cuda": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        """
        初始化特征提取器
        :param kwargs: 配置信息
        """
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()

    def generate(self):
        """载入模型与权值"""
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = FaceNet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def get_encoding(self, image):
        """
        获取人脸图像的编码
        :param image: 人脸图像
        :return: 人脸特征编码
        """
        with torch.no_grad():  # 关闭梯度计算
            # 调整图像大小
            image = ImageUtil.resize_image(
                image=image,
                size=(self.input_shape[1], self.input_shape[0]),
                letterbox_image=self.letterbox_image,
            )

            # 归一化
            image = ImageUtil.normalize(image)

            # 将通道调整至第0维
            image = np.expand_dims(image.transpose(2, 0, 1), 0)

            # 转换为张量
            photo = torch.from_numpy(image).type(torch.FloatTensor)

            if self.cuda:
                photo = photo.cuda()

            # 图片传入网络进行预测
            output = self.net(photo)[0].cpu().numpy()

        return output


