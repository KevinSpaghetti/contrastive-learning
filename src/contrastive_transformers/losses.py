import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.linalg import norm

class NormalizedCrossEntropyLoss(nn.Module):
    
    def __init__(self, temperature = 0.2, smoothing=0.0):
        super().__init__()
        
        self.temperature = temperature
        self.smoothing = smoothing
        self.sim = nn.CosineSimilarity(dim=-1)
    def forward(self, anchors, positives, negatives):
        
        anchor_positives = self.sim(anchors.unsqueeze(1), positives.unsqueeze(0)) / self.temperature
        anchor_negatives = self.sim(anchors.unsqueeze(1), negatives.unsqueeze(0)) / self.temperature
        
        anchor_positives_negatives = torch.cat([anchor_positives, anchor_negatives], 1)
        labels = torch.arange(anchors.shape[0], device=anchors.device)
        
        return F.cross_entropy(anchor_positives_negatives, labels, label_smoothing=self.smoothing).mean()

class SupConLoss(nn.Module):
    
    def __init__(self, temperature = 1, eps=1e-8):
        super().__init__()
        
        self.eps = eps
        self.temperature = temperature
    
    def pdist(self, A, B, dim=1):
        a_norm = A / A.norm(dim=dim)[:, None]
        b_norm = B / B.norm(dim=dim)[:, None]
        res = a_norm @ b_norm.mT  
        return res
    
    def forward(self, anchors, positives, labels):
        
        positive_mat = labels[:, None] == labels[None, :]

        norm = -1/torch.sum(positive_mat, axis=1)

        distances = torch.exp(self.pdist(anchors, positives) / self.temperature)
 
        sms = torch.zeros(anchors.shape[0], device=anchors.device)
        for i, row in enumerate(positive_mat):
            cols = row.nonzero()

            for col in cols:
                denom = torch.sum(distances[i] * ~positive_mat[i])
                sms[i] += torch.log(distances[i, col[0]] / (distances[i, col[0]] + denom))

        l_sup = torch.sum(norm * sms)
        return l_sup
