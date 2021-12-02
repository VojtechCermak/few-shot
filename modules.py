import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    '''
    Basic convolutional backbone - VGG style.
    '''
    def __init__(self, ch=1, D=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ch, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),     
       
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, D), # Outputs embedding
        )

    def forward(self, x):
        return self.main(x)


class BaseNetwork(pl.LightningModule):
    def __init__(self, backbone, loss):
        super().__init__()
        self.backbone = backbone  # x -> D (embedding)
        self.loss = loss          # D -> loss

    def embedding(self, x):
        return self.backbone(x)

    def forward(self, x, target):
        embedding = self.backbone(x)
        loss = self.loss(embedding, target)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self(*batch)
        return loss


class StandardNetwork(BaseNetwork):
    '''
    Basic network structure with batch and targets input. 
    Can be used for basic softmax, angular softmax and online mining methods.
    '''
    def forward(self, x, target):
        embedding = self.backbone(x)
        loss = self.loss(embedding, target)
        return loss


class TripletNetwork(BaseNetwork):
    '''
    Triplet network structure with input: anchor, positives and negatives.
    '''
    def forward(self, a, p, n):
        output_a = self.backbone(a)
        output_p = self.backbone(p)
        output_n = self.backbone(n)
        loss = self.loss(output_a, output_p, output_n)
        return loss


class SiameseNetwork(BaseNetwork):
    '''
    Siamese network structure with input: sample pair and positive/negative label.
    '''
    def forward(self, a, b, target):
        output_a = self.backbone(a)
        output_b = self.backbone(b)
        loss = self.loss(output_a, output_b, target)
        return loss


# TODO: Create validation module and add it there
'''
    from sklearn.neighbors import KNeighborsClassifier
    def validation_step(self, batch, batch_idx, loader_idx):
        x, y = batch
        embedding = self.embedding(x)
        return embedding, y

    def validation_epoch_end(self, outputs):
        # Get train embeddings
        x_train, y_train = [], []
        for embedding, label in outputs[0]:
            x_train.append(embedding)
            y_train.append(label)
        x_train, y_train = torch.cat(x_train), torch.cat(y_train)

        # Get test embeddings
        x_test, y_test = [], []
        for embedding, label in outputs[1]:
            x_test.append(embedding)
            y_test.append(label)
        x_test, y_test = torch.cat(x_test), torch.cat(y_test)

        # Train k-nn classifier
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(x_train.cpu().numpy(), y_train.cpu().numpy())

        # Accuracy on all test samples
        y_pred = classifier.predict(x_test.cpu().numpy())
        acc = sum(y_pred == y_test.cpu().numpy()) / len(y_pred)
        self.log('acc', acc)
        return x_test, y_test
'''