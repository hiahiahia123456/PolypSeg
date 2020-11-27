import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os

def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP = torch.Tensor([1])
    # recall
    Recall = TP / (TP + FN)
    # Specificity or true negative rate
    Specificity = TN / (TN + FP)
    # Precision or positive predictive value
    Precision = TP / (TP + FP)
    # F1 score = Dice
    F1 = 2 * Precision * Recall / (Precision + Recall)
    # F2 score
    F2 = 5 * Precision * Recall / (4 * Precision + Recall)
    # Overall accuracy
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)
    # IoU for poly
    IoU_poly = TP / (TP + FP + FN)
    # IoU for background
    IoU_bg = TN / (TN + FP + FN)
    # mean IoU
    IoU_mean = (IoU_poly + IoU_bg) / 2.0

    return Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean
