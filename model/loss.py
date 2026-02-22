import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
import math


def box_l1_loss(pred_boxes, target_boxes):
    return F.l1_loss(pred_boxes, target_boxes, reduction='mean')


def box_giou_loss(pred_boxes, target_boxes, eps=1e-7):
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = area_pred + area_target - inter + eps

    iou = inter / union

    x1_c = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_c = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_c = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_c = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    area_c = (x2_c - x1_c) * (y2_c - y1_c) + eps

    giou = iou - (area_c - union) / area_c
    giou = torch.clamp(giou, -1.0, 1.0)
    return 1 - giou.mean()