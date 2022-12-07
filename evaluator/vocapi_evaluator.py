import torch
from torch.autograd import Variable
from data.voc0712 import VOCDetection, VOC_CLASSES
import os
import time
import numpy as np
import pickle

import xml.etree.ElementTree as ET


class VOCAPIEvaluator():
    def __init__(self, data_root, img_size, device, transform, set_type='test', year='2007', display=False):
        self.data_root = data_root
        self.img_size = img_size
        self.device = device
        self.transform = transform
        self.labelmap = VOC_CLASSES
        self.set_type = set_type
        self.year = year
        self.display = display

        # path
        self.devkit_path = data_root + 'VOC' + year
        self.annopath = os.path.join(data_root, 'VOC2007', 'Annotations', '%s.xml')
        self.imgpath = os.path.join(data_root, 'VOC2007', 'JPEGImages', '%s.jpg')
        self.imgsetpath = os.path.join(data_root, 'VOC2007', 'ImageSets', 'Main', set_type+'.txt')
        self.output_dir = self.get_output_dir('voc_eval/', self.set_type)

        # dataset
        self.dataset = VOCDetection(root=data_root, 
                                    image_sets=[('2007', set_type)],
                                    transform=transform
                                    )

    def evaluate(self, net): ##目标检测模型的测试函数
        net.eval()
        num_images = len(self.dataset)
        self.all_boxes = [[[] for _ in range(num_images)]  ##为所有类别(labelmap)在每张图片上预测的边界框和得分创建一个空列表
                        for _ in range(len(self.labelmap))]
        ##[] for _ in range(num_images) 生成了num_images个[]，后面类似

        # timers
        det_file = os.path.join(self.output_dir, 'detections.pkl')

        for i in range(num_images):
            im, gt, h, w = self.dataset.pull_item(i) ##从数据集中获取一个样本

            x = Variable(im.unsqueeze(0)).to(self.device) ##将图像(im)转换为tensor
            t0 = time.time()
            # forward
            bboxes, scores, labels = net(x)
            detect_time = time.time() - t0
            scale = np.array([[w, h, w, h]])
            bboxes *= scale

            for j in range(len(self.labelmap)):
                inds = np.where(labels == j)[0]  ##类别是j的目标框在预测框中的下标
                if len(inds) == 0:
                    self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32) ##创建空数组
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_dets = np.hstack((c_bboxes,
                                    c_scores[:, np.newaxis])).astype(np.float32,
                                                                    copy=False) ##将包含物体边界框位置信息和置信度得分的数组水平堆叠
                ##c_scores[:, np.newaxis]是将c_scores转换为列向量
                self.all_boxes[j][i] = c_dets
                ##将当前图像中类别为j的检测框以及对应的置信度存储在self.all_boxes列表中，其中j是类别的编号，i是当前图像在数据集中的编号

            if i % 500 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
                ##im_detect: 501/10000 0.345s

        with open(det_file, 'wb') as f:  ##将目标检测结果保存在二进制文件中
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')  ##对检测结果进行评估
        self.evaluate_detections(self.all_boxes)

        print('Mean AP: ', self.map)
  

    def parse_rec(self, filename):  ##解析PASCAL VOC格式的XML文件
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects


    def get_output_dir(self, name, phase):  ##获取输出文件夹的路径的
        filedir = os.path.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir


    def get_voc_results_file_template(self, cls):  ##保存分类器cls检测结果的文本文件路径
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'det_' + self.set_type + '_%s.txt' % (cls)
        filedir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


    def write_voc_results_file(self, all_boxes):  ##将检测到的目标边界框信息写入到文件中
        for cls_ind, cls in enumerate(self.labelmap):
            if self.display:
                print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(cls)
            ##获取指定类别的结果文件名模板，模板包含了文件路径和文件名，其中文件名包含了类别名称和数据集集合类型
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.dataset.ids):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))
                        ##index[1] 表示图像的 ID，dets[k, -1] 表示检测结果的置信度，dets[k, 0] 到 dets[k, 3] 分别表示检测结果的左上角和右下角坐标


    def do_python_eval(self, use_07=True):  ##计算目标检测算法在验证集上的性能表现
        cachedir = os.path.join(self.devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        for i, cls in enumerate(self.labelmap):
            filename = self.get_voc_results_file_template(cls)  ##对应的 VOC 数据集检测结果文件的路径
            rec, prec, ap = self.voc_eval(detpath=filename, 
                                          classname=cls, 
                                          cachedir=cachedir,   ##cachedir参数是指缓存文件的路径
                                          ovthresh=0.5, 
                                          use_07_metric=use_07_metric
                                        )
            ##rec召回率 prec精度 ap平均精度
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)  ##将每个类别的precision-recall曲线和AP值保存到pickle文件中
        if self.display:  ##self.display=True，代码将输出每个类别的平均精度，最后输出平均mAP
            self.map = np.mean(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('--------------------------------------------------------------')
        else:  ##self.display=False，则仅输出平均mAP
            self.map = np.mean(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))


    def voc_ap(self, rec, prec, use_07_metric=True):  ##计算单个类别的平均精度
        if use_07_metric:  ##11点评估法
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            mrec = np.concatenate(([0.], rec, [1.])) ##在首尾添加0，1，确保在曲线的两端取到合理的precision值
            mpre = np.concatenate(([0.], prec, [0.]))

            ##计算正确的平均精度
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  ##将precision曲线变成一个单调不降的函数

            i = np.where(mrec[1:] != mrec[:-1])[0]  ##查找mrec数组中相邻元素不相等的位置

            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            ##计算每个召回率变化位置处的矩形面积，这个面积的高度为该位置处的精确率，宽度为该位置处的召回率变化量。最后将所有矩形面积加起来即可得到AP。
        return ap


    def voc_eval(self, detpath, classname, cachedir, ovthresh=0.5, use_07_metric=True):
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        # read list of images
        with open(self.imgsetpath, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        ##将每个图片的名称从txt文件中读取出来，并去除每个名称后面的空格，形成一个包含所有图片名称的列表
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames): ##i为图片在imagenames列表中的索引
                recs[imagename] = self.parse_rec(self.annopath % (imagename))
                ##parse_rec方法来解析相应的标注文件，最后将解析得到的标注信息存储到recs[imagename]中
                if i % 100 == 0 and self.display:
                    print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            # save
            if self.display:  ##打印出正在处理哪个图像
                print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:  ##将recs变量序列化并保存到文件中
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])  ##将所有目标的bounding box信息存储到数组 bbox 中。
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)  ##det列表用于表示对于给定类别和图像，检测器是否成功检测到了目标，r长度的FALSE
            npos = npos + sum(~difficult)  ##表示在所有的图像中，当前类别classname的目标的总数(去掉difficult的目标)
            class_recs[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:  ##这行代码是在判断变量 lines 是否为空列表，如果为空则返回 False

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  ##四个 bounding box 的坐标信息

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]  ##排序
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids) ##nd表示检测框数量
            tp = np.zeros(nd)  ##真阳性(tp)
            fp = np.zeros(nd)  ##假阳性(fp)
            for d in range(nd):
                R = class_recs[image_ids[d]]  ##表示图像中所有 类别为当前所计算精度和召回率的类别 的物体
                bb = BB[d, :].astype(float)  ##检测结果bounding box的位置信息
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)  ##R['bbox'] 是指 class_recs[image_ids[d]] 中对应目标物体的坐标框
                if BBGT.size > 0:
                    ixmin = np.maximum(BBGT[:, 0], bb[0])  ##真实框与预测框
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                        (BBGT[:, 2] - BBGT[:, 0]) *
                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)  ##最大值的索引

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:  ##判断GT框的难度是否为困难样本的代码
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap


    def evaluate_detections(self, box_list):
        self.write_voc_results_file(box_list)
        self.do_python_eval()


if __name__ == '__main__':
    pass