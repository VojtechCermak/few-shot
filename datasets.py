import numpy as np
import torch
from torch.utils.data import Dataset

class TupleDataset(Dataset):
    def __init__(self, dataset, targets):
        self.dataset = dataset
        self.targets = targets
        self.index = torch.tensor(np.arange(len(dataset)))

    def get_data(self, i):
        data, _ = self.dataset[i]
        return data

    def __len__(self):
        return len(self.dataset)


class RandomTriplets(TupleDataset):
    def __getitem__(self, i):
        '''
        Given anchor id, return ids of random triplet: anchor, positive, negative.
        '''
        anchor_target = self.targets[i]
        positive_set = self.index[(self.targets == anchor_target) & (self.index != i)]
        negative_set = self.index[(self.targets != anchor_target)]
        positive = np.random.choice(positive_set)
        negative = np.random.choice(negative_set)
        return self.get_data(i), self.get_data(positive), self.get_data(negative)


class RandomPairs(TupleDataset):
    def __getitem__(self, i):
        '''
        Given id, returns ids of positive or negative pair.
        '''
        target = self.targets[i]

        if np.random.rand() < 0.5:
            positive_set = self.index[(self.targets == target) & (self.index != i)]
            pair = np.random.choice(positive_set)
            category = 1.0
        else:
            positive_set = self.index[(self.targets != target)]
            pair = np.random.choice(positive_set)
            category = 0.0

        return self.get_data(i), self.get_data(pair), category


class ClassBalancedSampler():
    '''
    Sample K random samples for each of the P random classes.
    If weighted=True, class sampling is weighted by class freqeuncy.

    Example:
    loader = DataLoader(dataset, batch_sampler=ClassBalancedSampler(dataset.targets, P=3, K=10))
    '''
    def __init__(self, targets, P, K, weighted=True):
        self.weighted = weighted
        self.targets = targets
        self.P = P
        self.K = K
        self.no_batches = len(targets) // (P*K)
        self.index = np.arange(len(targets))
        self.class_unique, self.class_counts = np.unique(targets, return_counts=True)
        self.class_targets = {i: self.index[targets==i] for i in self.class_unique}

    def __iter__(self):
        if self.weighted:
            frequencies = self.class_counts / len(self.targets)
            categories = np.random.choice(self.class_unique, size=self.P, p=frequencies, replace=False)
        else:
            categories = np.random.choice(self.class_unique, size=self.P, replace=False)

        batches = []
        for _ in range(self.no_batches):
            batch = []
            for c in categories:
                batch.append(np.random.choice(self.class_targets[c], size=self.K, replace=False))
            batches.append(np.concatenate(batch))
        return iter(batches)
