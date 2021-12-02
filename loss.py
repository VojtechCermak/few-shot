import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftmaxLoss(nn.Module):
    '''
    Basic softmax loss: Linear layer + softmax + crossentropy
    Init: D - embedding size, C - number of classes
    Input: embedding, target
    '''
    def __init__(self, D, C):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.layer = nn.Sequential(
            #nn.ReLU(inplace=True), # TODO: add or not?
            nn.Linear(D, C)
            )
    
    def forward(self, embedding, target):
        return self.loss(self.layer(embedding), target)


class AngularSoftmaxLoss(nn.Module):
    '''
    Softmax loss variant based on ArcFace: https://arxiv.org/pdf/1703.07737.pdf
    Init: D - embedding size, C - number of classes
    Input: embedding, target
    '''
    def arccos(self, x, eps=1e-6):
        '''
        Smooth extension of arccos by linear function of arccos to whole R.
        '''
        arccos_derivative = lambda x: - 1 / torch.sqrt(1-x**2)
        c = torch.tensor(1 - eps)

        inner = torch.logical_and(x <= c, x >= -c)
        lower = x < -c
        upper = x >  c
        x = torch.where(inner, torch.acos(torch.clamp(x, min=-c, max=c)), x)
        x = torch.where(upper, torch.acos(c) + (x-c)*arccos_derivative(c), x)
        x = torch.where(lower, torch.acos(-c) + (x+c)*arccos_derivative(-c), x)
        return x


    def __init__(self, D, C, s, m1=1, m2=0, m3=0, eps=1e-6):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.W = nn.Parameter(torch.FloatTensor(C, D), requires_grad=True)
        self.W = nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        self.eps = eps
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.s = s

    def forward(self, embedding, target):
        W = F.normalize(self.W, p=2)
        x = F.normalize(embedding, p=2)
        logits = torch.matmul(x, W.T) # cos_theta (shape: BxC)
        
        # Add margins to target logits
        logit_target = logits.clone().gather(1, target.unsqueeze(1)) # shape: Bx1
        theta = self.arccos(logit_target)        
        logit_target = torch.cos(self.m1*theta + self.m2) - self.m3
        logits = self.s*logits.scatter_(1, target.unsqueeze(1), logit_target)
        return self.loss(logits, target)


class OnlineTripletLoss(nn.Module):
    '''
    Triplet loss with online batch-hard triplet mining strategy.
    Input: embedding, target
    '''
    def __init__(self, margin=1.0, squared=False, soft=False):
        super().__init__()
        self.loss = TripletLoss(margin, squared, soft)

    def batch_hard_triplets(self, embedding, target):
        '''
        For each anchor, get triplet of hardest positive and hardest negative
        '''
        with torch.no_grad():
            distances = torch.cdist(embedding, embedding)
            
        distances = distances.cpu().numpy()
        target = target.cpu().numpy()
        index = np.arange(len(target))

        positives = []
        negatives = []
        for i in range(len(target)):
            # TODO: This batch hard-hard strategy completely fails on MNIST. Bug?

            # Mask for positives
            mask = (target[i] == target) & (index != i) 
            mask_idx = distances[i][mask].argmax()
            positives.append(index[mask][mask_idx])

            # Mask for negatives
            mask = target[i] != target 
            mask_idx = distances[i][mask].argmin()
            negatives.append(index[mask][mask_idx])

        return embedding, embedding[positives], embedding[negatives]

    def forward(self, embedding, target):
        a, p, n = self.batch_hard_triplets(embedding, target)
        return self.loss(a, p, n)


class TripletLoss(nn.Module):
    """
    Triplet loss
    Input: achnor, positive, negative triplets.
    """

    def __init__(self, margin=1.0, squared=False, soft=False):
        super().__init__()
        self.margin = margin
        self.squared = squared
        self.soft = soft

    def distance(self, a, b):
        if self.squared:
            return torch.norm(a-b, p=2, dim=1)**2
        else:
            return torch.norm(a-b, p=2, dim=1)

    def max(self, a):
        if self.soft:
            return F.softplus(a)
        else:
            return F.relu(a)

    def forward(self, a, p, n):
        loss = self.max(self.distance(a, p) - self.distance(a, n) + self.margin)
        return loss.mean()


class SiameseLoss(nn.Module):
    def __init__(self, D):
        self.loss = nn.BCEWithLogitsLoss()
        self.layer = nn.Linear(D, 1)

    def forward(self, a, b, target):
        dist = torch.norm(a-b, p=2, dim=1) #TODO: Enable to use arbitrary metric
        return self.loss(self.layer(dist), target)


class ContrastiveSiameseLoss(nn.Module):
    pass # TODO


class OnlineSiameseLoss(nn.Module):
    pass # TODO