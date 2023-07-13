import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as F

from IPython.display import display
class FocalLoss(nn.Module):

    def __init__(self, gamma=3, alpha=0.75):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        ori_shp=target.shape
        logp = self.ce(input, target)
        #print("logp: ", logp)
        #p = torch.sigmoid(logp)
        #print("p1: ", p)
        p = torch.exp(-logp)
        #print("p2: ",p)
        loss = self.alpha * (1-p) ** self.gamma * logp
        return loss.view(ori_shp)

# # coding=utf-8
# import torch
# import torch.nn.functional as F
#
# from torch import nn
# from torch.nn import CrossEntropyLoss
# import numpy as np
#
#
# class FocalLoss(nn.Module):
#     """
#     Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
#     Args:
#         num_class: number of classes
#         alpha: class balance factor shape=[num_class, ]
#         gamma: hyper-parameter
#         reduction: reduction type
#     """
#
#     def __init__(self, num_class=2, alpha=None, gamma=2, reduction='none'):
#         super(FocalLoss, self).__init__()
#         self.num_class = num_class
#         self.gamma = gamma
#         self.reduction = reduction
#         self.smooth = 1e-4
#         self.alpha = alpha
#         if alpha is None:
#             self.alpha = torch.ones(num_class, ) - 0.5
#         elif isinstance(alpha, (int, float)):
#             self.alpha = torch.as_tensor([alpha] * num_class)
#         elif isinstance(alpha, (list, np.ndarray)):
#             self.alpha = torch.as_tensor(alpha)
#         if self.alpha.shape[0] != num_class:
#             raise RuntimeError('the length not equal to number of class')
#
#     def forward(self, logit, target):
#         """
#         N: batch size C: class num
#         :param logit: [N, C] 或者 [N, C, d1, d2, d3 ......]
#         :param target: [N] 或 [N, d1, d2, d3 ........]
#         :return:
#         """
#         # assert isinstance(self.alpha,torch.Tensor)\
#         alpha = self.alpha.to(logit.device)
#         prob = F.softmax(logit, dim=1)
#
#         if prob.dim() > 2:
#             # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
#             N, C = logit.shape[:2]
#             prob = prob.view(N, C, -1)
#             prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
#             prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
#
#         ori_shp = target.shape
#         target = target.view(-1, 1)
#         print(target.type(torch.int64))
#         prob = prob.gather(1, target.type(torch.int64).unsqueeze(1)).view(-1) + self.smooth  # avoid nan
#         logpt = torch.log(prob)
#         # alpha_class = alpha.gather(0, target.squeeze(-1))
#         alpha_weight = alpha[target.squeeze().long()]
#         loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt
#
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'none':
#             loss = loss.view(ori_shp)
#
#         return loss

