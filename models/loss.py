import torch
import torch.nn as nn
import numpy as np


class MSEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()

    def forward(self, logits, targets, mask):
        ##对logits进行sigmoid函数变换，并使用clamp函数将变换后的值限制在[1e-4, 1-1e-4]的范围内
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)


        pos_id = (mask==1.0).float()  ##pos_id表示正样本的掩码
        neg_id = (mask==0.0).float()  ##neg_id表示负样本的掩码
        pos_loss = pos_id * (inputs - targets)**2  ##所有正样本的损失
        neg_loss = neg_id * (inputs)**2  ##所有负样本的损失
        loss = 5.0*pos_loss + 1.0*neg_loss

        return loss


def iou_score(bboxes_a, bboxes_b):
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)  ##计算张量上dim=1上所有元素的乘积
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    ##en的作用是计算每个检测框的面积是否大于0，如果大于0则为1，否则为0。这个操作的目的是去掉宽或高为0的检测框，避免后续计算中出现除0的情况
    area_i = torch.prod(br - tl, 1) * en  ## 即重叠区域的面积
    return area_i / (area_a + area_b - area_i)


def compute_loss(pred_conf, pred_cls, pred_txtytwth, targets):
    batch_size = pred_conf.size(0)
    # 损失函数
    conf_loss_function = MSEWithLogitsLoss()
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    # 预测
    pred_conf = pred_conf[..., 0]           # 置信度 [B, HW,]
    pred_cls = pred_cls.permute(0, 2, 1)    # [B, C, HW]
    pred_txty = pred_txtytwth[..., :2]      # [B, HW, 2]
    pred_twth = pred_txtytwth[..., 2:]      # [B, HW, 2]

    # 标签  
    gt_conf = targets[..., 0].float()                 # [B, HW,]
    gt_obj = targets[..., 1].float()                  # [B, HW,]
    gt_cls = targets[..., 2].long()                   # [B, HW,]
    gt_txty = targets[..., 3:5].float()               # [B, HW, 2]
    gt_twth = targets[..., 5:7].float()               # [B, HW, 2]
    gt_box_scale_weight = targets[..., 7]             # 损失权重 [B, HW,]
    gt_mask = (gt_box_scale_weight > 0.).float()      # [B, HW,]

    # 置信度损失
    conf_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)
    conf_loss = conf_loss.sum() / batch_size
    
    # 类别损失
    cls_loss = cls_loss_function(pred_cls, gt_cls) * gt_mask
    cls_loss = cls_loss.sum() / batch_size
    
    # 边界框txty的损失
    txty_loss = txty_loss_function(pred_txty, gt_txty).sum(-1) * gt_mask * gt_box_scale_weight
    txty_loss = txty_loss.sum() / batch_size

    # 边界框twth的损失
    twth_loss = twth_loss_function(pred_twth, gt_twth).sum(-1) * gt_mask * gt_box_scale_weight
    twth_loss = twth_loss.sum() / batch_size
    bbox_loss = txty_loss + twth_loss

    #总的损失
    total_loss = conf_loss + cls_loss + bbox_loss

    return conf_loss, cls_loss, bbox_loss, total_loss


if __name__ == "__main__":
    pass
