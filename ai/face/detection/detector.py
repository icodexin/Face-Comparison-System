import torch.nn as nn

from ai.face.detection.models import RetinaFace
from ai.face.detection.utils import *
from ai.utils import ImageUtil
from utils.logger import logger


class FaceDetector:
    """人脸检测器"""
    # 工具参数
    _defaults = {
        # retinaface 训练完的权值路径
        "model_path": '../model_data/Retinaface_mobilenet0.25.pth',
        # retinaface 所使用的主干网络，有 mobilenet 和 resnet50
        "backbone": "mobilenet",
        # retinaface 中只有得分大于置信度的预测框会被保留下来
        "confidence": 0.9,
        # retinaface 中非极大抑制所用到的nms_iou大小
        "nms_iou": 0.45,
        # 是否需要进行图像大小限制。
        # 输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
        # 开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        # 会导致检测结果偏差，主干为resnet50不存在此问题。
        # 可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        "input_shape": [640, 640, 3],
        # 是否需要进行图像大小限制。
        "letterbox_image": False,
        # 是否使用Cuda
        "cuda": False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        """初始化FaceDetector检测器"""
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 不同主干网络的config信息
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        # 生成先验框
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=(self.input_shape[0], self.input_shape[1])).get_anchors()

        # 载入模型与权值
        self.generate()
        logger().info('Successfully initialized a Tool: Face Detector')

    def generate(self):
        """载入模型与权值"""
        self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()
        logger().info('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        logger().info(f'{self.model_path} model, and classes loaded.')

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image):
        """
        检测图片
        :param image: 输入图像
        :return: 未检测到人脸返回None
        """
        # 转换为numpy形式
        image = np.array(image, np.float32)
        # 计算输入图片的高和宽
        im_height, im_width, _ = image.shape
        # 计算scale，用于将获得的预测框转换为原图的高宽
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]
        #  letterbox_image可以给图像增加灰条，实现不失真的resize
        if self.letterbox_image:
            image = ImageUtil.letterbox_image(image, (self.input_shape[1], self.input_shape[0]))
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            # 图片预处理，归一化
            image = torch.from_numpy(ImageUtil.preprocess(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()

            #  传入网络进行预测
            loc, conf, landms = self.net(image)

            #  对预测框进行解码
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #  获得预测结果的置信度
            conf = conf.data.squeeze(0)[:, 1:2]
            #  对人脸关键点进行解码
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            #   对人脸识别结果进行堆叠
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            # 未检测到人脸
            if len(boxes_conf_landms) <= 0:
                return np.array([])

            # 如果使用了letterbox_image的话，要把灰条的部分去除掉
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.input_shape[0], self.input_shape[1]]),
                                                             np.array([im_height, im_width]))

        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        return boxes_conf_landms


if __name__ == "__main__":
    import cv2
    import os

    face_detector = FaceDetector()

    img_path = input('Input image filepath:')
    if not os.path.exists(img_path):
        print('No this image file!')
        exit(1)
    image = cv2.imread(img_path)
    if image is None:
        print('Open Error! Check your image file!')
        exit(1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes_conf_landms = face_detector.detect_image(image)

    for index, face in enumerate(boxes_conf_landms):
        print("人脸{}: 置信度{:.4f}".format(index, face[4]))
        text = "{:.4f}".format(face[4])
        b = list(map(int, face))

        # 标出人脸框和置信度
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        print(f"人脸框坐标[(左上_x, 左上_y), (右下_x, 右下_y)]:{(b[0], b[1])}, {(b[2], b[3])}")

        # 标出人脸关键点
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
        print(f"左眼坐标:{(b[5], b[6])}")
        print(f"右眼坐标:{(b[7], b[8])}")
        print(f"鼻尖坐标:{(b[9], b[10])}")
        print(f"左嘴角坐标:{(b[11], b[12])}")
        print(f"右嘴角坐标:{(b[13], b[14])}")
        print("------------------------------------------")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print("\n按下Q键退出程序")
    while True:
        cv2.imshow("Face Detection", image)
        # 按下 q 键退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
