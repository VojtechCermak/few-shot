import os
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from datasets import ClassBalancedSampler, RandomTriplets, RandomPairs


class CMNIST(MNIST):
    '''
    Subset MNIST by using only some classes.
    '''
    def __init__(self, classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(classes) == int:
            classes = np.arange(classes)
        assert len(classes) <= 10 and len(classes) > 0
        self.classes = classes
        select = np.isin(self.targets.cpu().numpy(), classes)
        self.data, self.targets = self.data[select], self.targets[select]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')


class MNISTModule(pl.LightningDataModule):
    def __init__(self, data_dir=".", sampler=None, P=5, batch_size=64, balanced=False):
        super().__init__()
        self.sampler = sampler
        self.balanced = balanced
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.P = P
        self.batch_size = batch_size

    def setup(self, stage=None):
        mnist_train = CMNIST(np.arange(10)[:self.P], self.data_dir, train=True, transform=self.transform)
        self.train = self.dataset_sampler(mnist_train)
        self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def dataset_sampler(self, dataset):
        if self.sampler is None:
            return dataset
        elif self.sampler == 'triplet':
            return RandomTriplets(dataset, dataset.targets)
        elif self.sampler == 'pair':
            return RandomPairs(dataset, dataset.targets)

    def train_dataloader(self):
        if self.balanced:
            sampler = ClassBalancedSampler(self.train.targets, P=self.P, K=(self.batch_size // self.P))
            return DataLoader(self.train, batch_sampler=sampler, num_workers=6)
        else:
            return DataLoader(self.train, batch_size=self.batch_size, num_workers=6)

    def val_dataloader(self):
        loader_base = DataLoader(self.mnist_train, batch_size=64, num_workers=6)
        loader_query = DataLoader(self.mnist_test, batch_size=64, num_workers=6)
        return [loader_base, loader_query]