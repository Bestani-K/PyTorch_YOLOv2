import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b): ##计算两个边界框box_a和box_b的交集面积
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)  ##clip函数将面积限制在0和无穷大之间
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):  ##计算两个边界框数组之间的Jaccard相似度
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):  ##将多个增强操作整合在一起使用。
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):  ##将图像从整数类型转换为浮点类型
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Normalize(object):  ##对图像进行归一化处理
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.  ##将其值范围从[0,255]缩放到[0,1]之间
        image -= self.mean
        image /= self.std

        return image, boxes, labels


class ToAbsoluteCoords(object):  ##将归一化的坐标转换为绝对坐标
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):  ##将图像中物体框的坐标转换为百分比坐标
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class ConvertColor(object):  ##将图像从一种颜色空间转换为另一种颜色空间
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class Resize(object):  ##调整图像大小的转换操作
    def __init__(self, size=640):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation(object):  ##随机饱和度增强的图像增强方法
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):  ##将所有大于360的像素值减去360
    def __init__(self, delta=18.0):  ##delta确定调整后饱和度和亮度值的范围的参数
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)  ##image[:, :, 0] 存储了图像中每个像素的色调信息
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):  ##随机增加图像颜色噪声
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            ##self.perms 是一个包含了6种可能通道交换方式的元组列表。
            ##random.randint 随机选择是否进行通道交换，如果选中了，则随机选择一个元组，表示将原图的RGB通道交换为对应的BGR、BRG、GBR等通道
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class RandomContrast(object):  ##调整图像对比度的随机变换
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):  ##生成一个随机整数，其值为0或1
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):  ##将输入的图像的亮度值进行随机增加或减少
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomSampleCrop(object):  ##对图像进行随机采样裁剪的类
    def __init__(self):
        self.sample_options = (  ##6种不同的随机剪裁模式
            None,  ##使用整个原始输入图像。
            (0.1, None),  ##在与至少一个边界框最小IoU为0.1的情况下采样一个裁剪区域。
            (0.3, None),  ##在与至少一个边界框最小IoU为0.3的情况下采样一个裁剪区域。
            (0.7, None),  ##在与至少一个边界框最小IoU为0.7的情况下采样一个裁剪区域。
            (0.9, None),  ##在与至少一个边界框最小IoU为0.9的情况下采样一个裁剪区域。
            (None, None),  ##随机采样一个裁剪区域，不考虑边界框的IoU。
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            sample_id = np.random.randint(len(self.sample_options)) ##随机选择裁剪模式
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)  ##区间 (0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)  ##生成crop区域的左上角坐标
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # 在输入的图像中每个物体框与当前随机选取的样本矩形（rect）之间的Jaccard相似度
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # 根据上一步得到的 rect 对图像进行裁剪
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]
                ##rect[1] 和 rect[3] 对应裁剪后图像的高度，rect[0] 和 rect[2] 对应裁剪后图像的宽度

                ## 计算bounding box的中心坐标
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # 表示每个 bounding box 的中心点是否在裁剪后的区域内
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # 每个框的中心是否在采样后的矩形区域中
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # m1用于判断x轴坐标的限制条件是否满足，m2用于判断y轴坐标的限制条件是否满足
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                ## 选出中心点在采样后矩形区域中的bounding box，将这些bounding box复制到一个新的变量current_boxes中
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask] ##mask布尔值

                # 将采样后的矩形区域左上角坐标和当前框的左上角坐标比较，取其中较大的值作为当前框的左上角坐标
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])

                # 将框的左上角坐标与采样后的矩形的左上角对齐，即将左上角的坐标设置为(0,0)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])

                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    ##通过在原始图像的周围添加一些背景区域，从而扩大图像的尺寸，增加样本的多样性。
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        ##扩展后的图像能够完全包含原始图像

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        ##用mean值填充expand_image的所有像素
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        ##将原始图像复制到expand_image的随机位置上
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        ##对bounding boxes进行扩展操作

        return image, boxes, labels


class RandomMirror(object):  ##实现随机图像水平翻转的功能
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2): ##如果随机生成的整数为1（即50%的概率）
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2] ##将边框左右两个顶点的坐标位置进行互换
            ##2::-2表示从下标2开始到数组开头，以步长为-2的方式切片。
        return image, boxes, classes


class SwapChannels(object): ##将图像的通道交换位置的操作
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):  ##对输入的图像执行一系列色彩变换操作，以增加数据集的多样性，提高模型的鲁棒性
    def __init__(self):
        self.pd = [
            RandomContrast(),  ##随机调整图像对比度
            ConvertColor(transform='HSV'),  ##指定了颜色空间的转换方式
            RandomSaturation(),  ##增加图像数据的多样性
            RandomHue(),  ##改变图像色调
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)  ##对输入的图像亮度进行随机扰动
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels


class Augmentation(object):
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),             # 将int类型转换为float32类型
            ToAbsoluteCoords(),            # 将归一化的相对坐标转换为绝对坐标
            PhotometricDistort(),          # 图像颜色增强
            Expand(self.mean),             # 扩充增强
            RandomSampleCrop(),            # 随机剪裁
            RandomMirror(),                # 随机水平镜像
            ToPercentCoords(),             # 将绝对坐标转换为归一化的相对坐标
            Resize(self.size),             # resize操作
            Normalize(self.mean, self.std) # 图像颜色归一化
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class BaseTransform: ##数据预处理的基础变换类
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        # resize
        image = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        # normalize
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels
