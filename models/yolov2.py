import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import Conv, reorg_layer
from backbone import build_backbone

import numpy as np
from .loss import iou_score, compute_loss


class YOLOv2(nn.Module):
    def __init__(self,
                 cfg,  ##定义网络的超参数和结构
                 device,
                 input_size=416,
                 num_classes=20,
                 trainable=False, ##指定网络是否可训练，如果为 True，则所有参数都可以被优化更新
                 conf_thresh=0.001,  ##置信度阈值
                 nms_thresh=0.6,   ##非极大值抑制的阈值
                 topk=100,  ##保留置信度最高的前 K 个检测结果
                 anchor_size=None):
        super(YOLOv2, self).__init__()  ##该部分建议对照着网络结构图进行解析  https://ethereon.github.io/netscope/#/gist/d08a41711e48cf111e330827b1279c31
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = cfg['stride']
        self.topk = topk

        # Anchor box config
        self.anchor_size = torch.tensor(anchor_size)  # [KA, 2]
        self.num_anchors = len(anchor_size)
        self.anchor_boxes = self.create_grid(input_size)

        # 主干网络
        self.backbone, feat_dims = build_backbone(cfg['backbone'], cfg['pretrained'])
        ##cfg['pretrained'] 表示网络配置文件中指定的预训练模型的路径或权重文件的路径，如果为 None 则表示不使用预训练模型。
        
        # 检测头
        self.convsets_1 = nn.Sequential(
            Conv(feat_dims[-1], cfg['head_dim'], k=3, p=1),
            Conv(cfg['head_dim'], cfg['head_dim'], k=3, p=1)
        )

        # 融合高分辨率的特征信息
        self.route_layer = Conv(feat_dims[-2], cfg['reorg_dim'], k=1)
        self.reorg = reorg_layer(stride=2)

        # 检测头
        self.convsets_2 = Conv(cfg['head_dim']+cfg['reorg_dim']*4, cfg['head_dim'], k=3, p=1)
        ##cfg['reorg_dim']*4 即经过reorg层后的4张特征图，每张规格64
        
        # 预测层
        self.pred = nn.Conv2d(cfg['head_dim'], self.num_anchors*(1 + 4 + self.num_classes), 1)


        if self.trainable:
            self.init_bias()


    def init_bias(self):
        # init bias
        ##在训练初始阶段，Sigmoid 函数输出的值都在接近 0 或 1 的较小区间，导致梯度变化较小，使得训练难以进行。因此，需要对偏置进行适当的初始化，
        ##以便让 Sigmoid 函数在训练开始时输出的值分布在一个较合适的区间，这里的做法是将偏置设置为一个特定的值，使得 Sigmoid 函数在初始时的输出分布在接近 0.5 的区间。
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :self.num_anchors], bias_value)  ##constant_将张量初始化为value
        ##self.num_anchors 个偏置项表示锚框预测的物体是否存在（即代表物体的锚框的类别概率的偏置项）
        nn.init.constant_(self.pred.bias[..., 1*self.num_anchors:(1+self.num_classes)*self.num_anchors], bias_value)
        ##self.num_anchors 个偏置项表示锚框预测的物体的类别（即坐标偏移量对应的偏置项）
        ##偏置张量的前num_anchors个值对应了每个锚框的置信度预测偏置（confidence bias），接下来的4 * num_anchors个值对应了每个锚框的位置预测偏置（bbox regression bias），
        ##最后的num_classes * num_anchors个值对应了每个锚框的类别预测偏置（class bias）
        ##前num_anchors个是置信度，接下来num_anchors*classes是类别，再接下来num_anchors*4是关于坐标的偏移。


    def create_grid(self, input_size):
        w, h = input_size, input_size
        # 生成G矩阵
        fmp_w, fmp_h = w // self.stride, h // self.stride
        ##特征图宽高
        grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])  ##网格点的y坐标和x坐标，其中meshgrid函数可以用于生成一个网格状的坐标系

        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
        ##将 grid_x 和 grid_y 沿着最后一个维度进行拼接，并将结果的维度变为 (fmp_h * fmp_w, 2)
        ##其中第一列代表每个格子的 x 坐标，第二列代表每个格子的 y 坐标。这个 grid_xy 后面会用来计算预测框的中心坐标。

        grid_xy = grid_xy[:, None, :].repeat(1, self.num_anchors, 1)
        ##扩展成每个anchor对应一个格子的形式(fmp_w * fmp_h, 2) --> (fmp_w * fmp_h, num_anchors, 2)

        anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)
        ##(num_anchors, 2) --> anchor_size[None, :, :] --> (fmp_h * fmp_w, num_anchors, 2)

        anchor_boxes = torch.cat([grid_xy, anchor_wh], dim=-1)
        ##anchor_boxes : (fmp_h * fmp_w, num_anchors, 4)
        anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)
        ##anchor_boxes : (num_anchors * fmp_h * fmp_w, 4)

        return anchor_boxes        


    def set_grid(self, input_size):
        self.input_size = input_size
        self.anchor_boxes = self.create_grid(input_size)


    def decode_boxes(self, anchors, txtytwth_pred):
        # 获得边界框的中心点坐标和宽高
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[..., :2]) + anchors[..., :2]
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * anchors[..., 2:]
        ##tw与th表示的是特征图的缩放因子

        xywh_pred = torch.cat([xy_pred, wh_pred], -1) * self.stride
        ##生成anchor box的时候进行了缩小，现在需要将其放大回去，得到预测的bounding box在原图中的坐标。

        # 将中心点坐标和宽高换算成边界框的左上角点坐标和右下角点坐标
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1x2y2_pred[..., :2] = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred[..., 2:] = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        
        return x1y1x2y2_pred


    def nms(self, bboxes, scores):
        x1 = bboxes[:, 0]  #xmin
        y1 = bboxes[:, 1]  #ymin
        x2 = bboxes[:, 2]  #xmax
        y2 = bboxes[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            ##将置信度最高的那一个与所有的进行比较，xx1是一个数组
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            # 滤除超过nms阈值的检测框
            inds = np.where(iou <= self.nms_thresh)[0] ##找到IOU小于nms_thresh的边界框的下标
            order = order[inds + 1] ##将 order 数组中下标在 inds 中的所有元素删除，并返回删除后的新数组
            ##iou的长度比order长度少1

        return keep


    def postprocess(self, conf_pred, cls_pred, reg_pred):
        ##conf_pred表示预测的目标存在概率（N，1，H，W）, cls_pred表示预测的每个目标类别的概率（N，C，H，W）, reg_pred预测的目标框的位置信息（N，4，H，W）
        ##每个 anchor 对应的预测偏移量都存储在 reg_pred 的第二个维度
        anchors = self.anchor_boxes

        scores = (torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)).flatten()  ##每个类别对应的目标存在概率

        ##限制处理的框的数量
        num_topk = min(self.topk, reg_pred.size(0))

        #对置信度进行排序
        predicted_prob, topk_idxs = scores.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        #过滤掉置信度低的框
        keep_idxs = topk_scores > self.conf_thresh
        scores = topk_scores[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]

        ##topk_idxs = anchor_idx * num_classes + label_idx，目的是拿到anchor的索引，和其所对应了哪一个类别
        anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor') ##用topk_idxs逐一除以self.num_classes，用向下舍入
        labels = topk_idxs % self.num_classes

        reg_pred = reg_pred[anchor_idxs]  ##预测框坐标 (num_anchors, 4)
        anchors = anchors[anchor_idxs]  ##anchor坐标

        # 转化为左上角与右下角坐标，bboxes的形状是(N, num_topk, 4)
        bboxes = self.decode_boxes(anchors, reg_pred)
        
        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # 归一化边界框
        bboxes = bboxes / self.input_size
        bboxes = np.clip(bboxes, 0., 1.)  ##np.clip() 函数，将所有坐标限制在 [0, 1] 的范围内

        return bboxes, scores, labels


    @torch.no_grad() ##禁用梯度跟踪
    def inference(self, x):  ##仅用于预测
        # backbone主干网络
        feats = self.backbone(x)
        c4, c5 = feats['c4'], feats['c5']

        # 处理c5特征
        p5 = self.convsets_1(c5)

        # 融合c4特征
        p4 = self.reorg(self.route_layer(c4))
        p5 = torch.cat([p4, p5], dim=1)

        # 处理p5特征
        p5 = self.convsets_2(p5)

        # 预测
        prediction = self.pred(p5)

        B, abC, H, W = prediction.size()  ##(B, (num_classes+5)*num_anchors, H, W)
        KA = self.num_anchors
        NC = self.num_classes

        ##(B, abC, H, W) --> (B, H, W, abC) --> (B, H*W, abC)
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, -1, abC)

        ##(B, H * W * num_anchors, 1)
        conf_pred = prediction[..., :KA].contiguous().view(B, -1, 1)

        ##(B, H*W*num_anchors, num_classes)
        cls_pred = prediction[..., 1*KA : (1+NC)*KA].contiguous().view(B, -1, NC)

        ##(B, H*W*num_anchors, 4)
        txtytwth_pred = prediction[..., (1+NC)*KA:].contiguous().view(B, -1, 4)

        conf_pred = conf_pred[0]            #(H*W*num_anchors, 1)
        cls_pred = cls_pred[0]              #(H*W*num_anchors, num_classes)
        txtytwth_pred = txtytwth_pred[0]    #(H*W*num_anchors, 4)

        # 后处理
        bboxes, scores, labels = self.postprocess(conf_pred, cls_pred, txtytwth_pred)

        return bboxes, scores, labels


    def forward(self, x, targets=None): ##预测加训练
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone主干网络
            feats = self.backbone(x)
            c4, c5 = feats['c4'], feats['c5']

            # 处理c5特征
            p5 = self.convsets_1(c5)

            # 融合c4特征
            p4 = self.reorg(self.route_layer(c4))
            p5 = torch.cat([p4, p5], dim=1)

            # 处理p5特征
            p5 = self.convsets_2(p5)

            # 预测
            prediction = self.pred(p5)

            B, abC, H, W = prediction.size()
            KA = self.num_anchors
            NC = self.num_classes

            ##(B, H*W, abC)
            prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

            ##(B, H * W * num_anchors, 1)
            conf_pred = prediction[..., :KA].contiguous().view(B, -1, 1)

            ##(B, H*W*num_anchors, num_classes)
            cls_pred = prediction[..., 1*KA : (1+NC)*KA].contiguous().view(B, -1, NC)

            ##(B, H*W*num_anchors, 4)
            txtytwth_pred = prediction[..., (1+NC)*KA:].contiguous().view(B, -1, 4)  

            ## x1y1x2y2_pred : (num_anchors * num_samples, 4)
            x1y1x2y2_pred = (self.decode_boxes(self.anchor_boxes, txtytwth_pred) / self.input_size).view(-1, 4)
            ##用于存储所有目标的真实边界框坐标,targets[:, :, 7:] 用于从 targets tensor 中选取所有目标的坐标信息
            x1y1x2y2_gt = targets[:, :, 7:].view(-1, 4)

            # 计算预测框和真实框之间的IoU
            ##(batch_size * n_anchors, n_targets) --> (batch_size, n_anchors, n_targets)
            iou_pred = iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)

            # 将IoU作为置信度的学习目标,上下文管理器，使得其中的操作不会被反向传播，即这部分操作不会对模型的梯度产生影响
            with torch.no_grad():
                gt_conf = iou_pred.clone()
            
            # 将IoU作为置信度的学习目标 
            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            targets = torch.cat([gt_conf, targets[:, :, :7]], dim=2)

            # 计算损失
            (
                conf_loss,
                cls_loss,
                bbox_loss,
                total_loss
            ) = compute_loss(
                pred_conf=conf_pred, 
                pred_cls=cls_pred,
                pred_txtytwth=txtytwth_pred,
                targets=targets,
                )

            return conf_loss, cls_loss, bbox_loss, total_loss
