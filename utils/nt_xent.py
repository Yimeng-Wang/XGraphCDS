import torch
import numpy as np


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Sigmoid()
        #self.mask_samples_from_same_repr = self._get_correlated_mask(x).type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self,x):
        self.batch_size=x.shape[0]
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        #print('p1: ',zis)
        #print(zjs.shape)
        #print(representations)
        similarity_matrix = self.similarity_function(representations, representations)
        #print(similarity_matrix.shape)
        self.batch_size=zjs.shape[0]
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        #print('l_pos: ',l_pos)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        #print('r_pos: ',r_pos)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        self.mask_samples_from_same_repr = self._get_correlated_mask(zis).type(torch.bool)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        #print('loss: ',loss)
        return loss / (2 * self.batch_size)

class Align_score(torch.nn.Module):
    def __init__(self, device, batch_size):
        super(Align_score, self).__init__()
        self.batch_size = batch_size
        self.device = device
    def forward(self,zis, zjs):
        align_loss = (zis - zjs).norm(p=2, dim=1).pow(2).mean()
        #print(align_loss)
        return align_loss
        


class Uniform_score(torch.nn.Module):
    def __init__(self, device, batch_size):
        super(Uniform_score, self).__init__()
        self.batch_size = batch_size
        self.device = device
    def forward(self,zis, zjs, t=2):
        uniform_loss1=torch.pdist(zis, p=2).pow(2).mul(-t).exp().mean().log()
        uniform_loss2 = torch.pdist(zjs, p=2).pow(2).mul(-t).exp().mean().log()
        uniform_loss=(uniform_loss1 + uniform_loss2) / 2
        #print(uniform_loss)
        return uniform_loss
    