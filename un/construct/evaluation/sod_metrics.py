import torch
import numpy as np

def _eval_e(y_pred, y, num):
    #print(y.shape) torch.Size([1, 270, 480])
    _, h, w = y.shape
    pred = y_pred.expand(num, h, w)
    gt = y.expand(num, h, w)
    thlist = torch.linspace(0, 1 - 1e-10, num).reshape(num, 1)
    mask = thlist.expand(num, h * w).reshape(num, h, w)
    pred_threshold = torch.where(pred >= mask, 1, 0).float()
    fm = pred_threshold - torch.mean(pred_threshold, dim=(1, 2)).reshape(num, 1).expand(num, h * w).reshape(num, h, w)
    gt = gt.float()
    gt = gt - torch.mean(gt, dim=(1, 2)).reshape(num, 1).expand(num, h * w).reshape(num, h, w)
    align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
    enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
    score = torch.sum(enhanced, dim=(1, 2)) / (y.numel() - 1 + 1e-20)
    return score


def _eval_pr(y_pred, y, num):
    _, h, w = y.shape
    pred = y_pred.expand(num, h, w)
    gt = y.expand(num, h, w)
    #thlist = torch.linspace(0, 1 - 1e-10, num).cuda().reshape(num, 1)
    thlist = torch.linspace(0, 1 - 1e-10, num).reshape(num, 1)
    mask = thlist.expand(num, h * w).reshape(num, h, w)
    pred_threshold = torch.where(pred >= mask, 1, 0).float()
    tp = torch.sum(pred_threshold * gt, dim=(1, 2))
    prec, recall = tp / (torch.sum(pred_threshold, dim=(1, 2)) + 1e-20), tp / (torch.sum(gt, dim=(1, 2)) + 1e-20)
    return prec, recall


def _S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.float().mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q


def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.float().mean()
    sigma_x = temp.float().std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score


def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    # print(Q)
    return Q


def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        #X = torch.eye(1).cuda() * round(cols / 2)
        X = torch.eye(1) * round(cols / 2)
        #Y = torch.eye(1).cuda() * round(rows / 2)
        Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        #i = torch.from_numpy(np.arange(0, cols)).cuda().float()
        i = torch.from_numpy(np.arange(0, cols)).float()
        #j = torch.from_numpy(np.arange(0, rows)).cuda().float()
        j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)
    return X.long(), Y.long()


def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.float().mean()
    y = gt.float().mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q