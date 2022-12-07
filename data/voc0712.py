import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES)))) ##将VOC_CLASSES中的类别名称和它们在列表中的索引一一对应，并创建一个新的字典。
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1  ##判断标注框是否为难识别的物体
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()  ##转换成小写字母并去掉两端的空格
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):  ##获取目标物体的边界框信息
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]

        return res


class VOCDetection(data.Dataset):
    def __init__(self,
                 root,
                 img_size=None,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, 
                 target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'
                 ):
        self.root = root
        self.img_size = img_size
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt


    def __len__(self):
        return len(self.ids)


    def pull_item(self, index):  ##返回在给定的索引处的数据
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot() ##解析数据集中对应图像的 XML 标注文件，并返回标注文件的根元素
        img = cv2.imread(self._imgpath % img_id)  ##读取图像
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)  ##将原始的VOC数据集的annotation转换为训练目标

        if self.transform is not None:
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            ##将输入的img和target的bbox传入transform中进行数据增强
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


    def pull_image(self, index):  ##返回指定索引的图像
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):  ##获取指定索引的图像的注释信息
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    from transform import Augmentation, BaseTransform

    img_size = 640
    pixel_mean = (0.406, 0.456, 0.485)
    pixel_std = (0.225, 0.224, 0.229)
    data_root = 'D:/pycharm/dataset/VOCdevkit/VOCdevkit'
    transform = Augmentation(img_size, pixel_mean, pixel_std)
    transform = BaseTransform(img_size, pixel_mean, pixel_std)

    # dataset
    dataset = VOCDetection(
        root=data_root,
        img_size=img_size, 
        image_sets=[('2007', 'trainval')],
        transform=transform
        )

    for i in range(1000):
        im, gt, h, w = dataset.pull_item(i)

        # to numpy
        image = im.permute(1, 2, 0).numpy()
        # to BGR
        image = image[..., (2, 1, 0)]
        # denormalize
        image = (image * pixel_std + pixel_mean) * 255
        ##模型训练时输入的图像经过了归一化处理，即减去均值(pixel_mean)并除以标准差(pixel_std)，这样做是为了使得各个通道的取值范围接近，有利于模型的训练。
        # to
        image = image.astype(np.uint8).copy()
        ##将图像像素值类型转换为 np.uint8 类型

        # draw bbox
        for box in gt:  ##将ground truth框可视化在图片上
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
            ##(int(xmin), int(ymin)) 表示矩形框的左上角点坐标，(int(xmax), int(ymax)) 表示矩形框的右下角点坐标，(0,0,255) 表示矩形框的颜色，这里是红色，(2) 表示矩形框的边框宽度
        cv2.imshow('gt', image)
        cv2.waitKey(0)
