import numpy as np
import random
import argparse
import os
import sys
sys.path.append('..')

from data.voc0712 import VOCDetection
from data.coco import COCODataset


def parse_args(): ##解析命令行参数
    parser = argparse.ArgumentParser(description='kmeans for anchor box')
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, widerface, crowdhuman')
    parser.add_argument('-na', '--num_anchorbox', default=5, type=int, ##默认的锚框数量是 5
                        help='number of anchor box.')
    parser.add_argument('-size', '--img_size', default=416, type=int,  ##图片默认大小是416
                        help='input size.')
    parser.add_argument('-ab', '--absolute', action='store_true', default=False,  ##控制在训练过程中是否使用绝对坐标。
                        help='absolute coords.')
    return parser.parse_args()
                    
args = parse_args()
                    

class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def iou(box1, box2):
    x1, y1, w1, h1 = box1.x, box1.y, box1.w, box1.h
    x2, y2, w2, h2 = box2.x, box2.y, box2.w, box2.h

    S_1 = w1 * h1
    S_2 = w2 * h2

    xmin_1, ymin_1 = x1 - w1 / 2, y1 - h1 / 2
    xmax_1, ymax_1 = x1 + w1 / 2, y1 + h1 / 2
    xmin_2, ymin_2 = x2 - w2 / 2, y2 - h2 / 2
    xmax_2, ymax_2 = x2 + w2 / 2, y2 + h2 / 2

    I_w = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
    I_h = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
    if I_w < 0 or I_h < 0:
        return 0
    I = I_w * I_h

    IoU = I / (S_1 + S_2 - I)

    return IoU


def init_centroids(boxes, n_anchors):  ##初始化聚类中心，n_anchors为需要聚类的数量，n_anchors是手动设置的
    centroids = []
    boxes_num = len(boxes)

    centroid_index = int(np.random.choice(boxes_num, 1)[0])
    ##这行代码从boxes中随机选择一个box作为第一个聚类中心点。boxes_num是所有boxes的数量。
    centroids.append(boxes[centroid_index])
    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0, n_anchors-1): ##创建n_anchors个聚类中心
        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance * np.random.random() ##生成[0.0, 1.0)

        for i in range(0, boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break
    return centroids  ##centroids 中的元素就是通过 K-means 聚类算法产生的聚类中心，即 anchor boxes 的值。


def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []

    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))  ##x,y,w,h
    
    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):  ##计算新的簇中心（即新的anchors）的宽和高
        new_centroids[i].w /= max(len(groups[i]), 1)
        new_centroids[i].h /= max(len(groups[i]), 1)

    return new_centroids, groups, loss# / len(boxes)


def anchor_box_kmeans(total_gt_boxes, n_anchors, loss_convergence, iters, plus=True):
    ##loss_convergence：迭代停止的阈值; iters：最大迭代次数; plus：是否使用k-means++算法。
    boxes = total_gt_boxes
    centroids = []
    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:  ##随机初始化锚框
        total_indexs = range(len(boxes))
        sample_indexs = random.sample(total_indexs, n_anchors)  ##从total_indexs序列中随机选择n_anchors个整数
        for i in sample_indexs:
            centroids.append(boxes[i])

    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while(True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids) ##每次do_kmeans都会更新质心
        iterations += 1
        print("Loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iters:
            break
        old_loss = loss

        for centroid in centroids:
            print(centroid.w, centroid.h)
    
    print("k-means result : ") 
    for centroid in centroids:
        if args.absolute:  ##是否要输出绝对宽度和高度
            print("w, h: ", round(centroid.w, 2), round(centroid.h, 2),
                "area: ", round(centroid.w, 2) * round(centroid.h, 2))
        else: ##宽度和高度需要除以网格的尺寸
            print("w, h: ", round(centroid.w / 32, 2), round(centroid.h / 32, 2),
                "area: ", round(centroid.w / 32, 2) * round(centroid.h / 32, 2))

    return centroids


if __name__ == "__main__":

    n_anchors = args.num_anchorbox
    img_size = args.img_size
    
    loss_convergence = 1e-6
    iters_n = 1000
    
    boxes = []
    if args.dataset == 'voc':
        data_root = os.path.join(args.root, 'VOCdevkit')
        dataset = VOCDetection(root=data_root,img_size=img_size)

        # VOC
        for i in range(len(dataset)):
            if i % 5000 == 0:
                print('Loading voc data [%d / %d]' % (i+1, len(dataset)))

            # For VOC
            img, _ = dataset.pull_image(i)
            w, h = img.shape[1], img.shape[0]
            _, annotation = dataset.pull_anno(i)

            # prepare bbox datas
            for box_and_label in annotation:
                box = box_and_label[:-1]
                xmin, ymin, xmax, ymax = box
                bw = (xmax - xmin) / max(w, h) * img_size
                bh = (ymax - ymin) / max(w, h) * img_size
                # check bbox
                if bw < 1.0 or bh < 1.0:
                    continue
                boxes.append(Box(0, 0, bw, bh))
            break

    elif args.dataset == 'coco':
        data_root = os.path.join(args.root, 'COCO')
        dataset = COCODataset(data_dir=data_root, img_size=img_size)

        for i in range(len(dataset)):
            if i % 5000 == 0:
                print('Loading coco datat [%d / %d]' % (i+1, len(dataset)))

            # For COCO
            img, _ = dataset.pull_image(i)
            w, h = img.shape[1], img.shape[0]
            annotation = dataset.pull_anno(i)

            # prepare bbox datas
            for box_and_label in annotation:
                box = box_and_label[:-1]
                xmin, ymin, xmax, ymax = box
                bw = (xmax - xmin) / max(w, h) * img_size
                bh = (ymax - ymin) / max(w, h) * img_size
                # check bbox
                if bw < 1.0 or bh < 1.0:
                    continue
                boxes.append(Box(0, 0, bw, bh))

    print("Number of all bboxes: ", len(boxes))
    print("Start k-means !")
    centroids = anchor_box_kmeans(boxes, n_anchors, loss_convergence, iters_n, plus=True)
