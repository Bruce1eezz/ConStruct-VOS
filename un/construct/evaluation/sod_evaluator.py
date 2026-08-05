import os
import time

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

class Eval_thread():
    def __init__(self, loader, method, dataset):
        self.loader = loader
        self.method = method
        self.dataset = dataset

    def run(self):
        print('eval: {} dataset with {} method.'.format(self.dataset, self.method))
        start_time = time.time()
        beta2 = 0.3
        alpha = 0.5
        mae_dict = dict()
        F_dict = dict()
        E_dict = dict()
        S_dict = dict()
        with torch.no_grad():
            for v_name, preds, gts in tqdm(self.loader):
                preds = preds.cuda()
                gts = gts.cuda()

                ####### MAE ######
                mean = torch.abs(preds - gts).mean()
                assert mean == mean, "mean is NaN"  # for Nan
                mae_dict[v_name] = mean

                # F Measure Score
                f_score = 0
                # E Measure Score
                e_score = torch.zeros(256).cuda()
                # S Measure Score
                sum_Q = 0
                for pred, gt in zip(preds, gts):
                    # F-Measure
                    prec, recall = self._eval_pr(pred, gt, 256)
                    f_score += (1 + beta2) * prec * recall / (beta2 * prec + recall+1e-10)
                    assert (f_score == f_score).all()  # for Nan
                    # E-Measure
                    e_score += self._eval_e(pred, gt, 256)
                    # S-Measure
                    y = gt.mean()
                    if y < 1e-4:
                        x = pred.mean()
                        Q = 1.0 - x
                    elif y == 1:
                        x = pred.mean()
                        Q = x
                    else:
                        gt[gt >= 0.5] = 1
                        gt[gt < 0.5] = 0
                        Q = alpha * self._S_object(pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                        if Q.item() < 0:
                            Q = torch.FloatTensor([0.0])[0].cuda()
                    assert Q==Q,'Q is NaN'
                    sum_Q += Q

                # F-Measure
                f_score /= len(preds)
                F_dict[v_name] = f_score
                # E-Measure
                e_score /= len(preds)
                E_dict[v_name] = e_score
                # S-Measure
                S_video = sum_Q / len(preds)
                S_dict[v_name] = S_video
            # MAE
            MAE_videos_max = torch.mean(torch.tensor(list(mae_dict.values()))).item()
            # Max F-Measure
            F_videos = torch.stack(list(F_dict.values())).mean(dim=0)
            F_videos_max = F_videos.max().item()
            # Max E-Measure
            E_videos = torch.stack(list(E_dict.values())).mean(dim=0)
            E_videos_max = E_videos.max().item()
            # S-Measure
            S_videos_mean = torch.mean(torch.tensor(list(S_dict.values()))).item()

            return '[cost:{:.2f}s] {} dataset with {} method get {:.3f} MAE, {:.3f} max F-measure, {:.3f} max E-measure, {:.3f} S-measure..'.format(
                time.time() - start_time, self.dataset, self.method, MAE_videos_max, F_videos_max, E_videos_max, S_videos_mean)
