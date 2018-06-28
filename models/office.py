"""ADDA model for Office dataset."""

import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.restored = False

        model_resnet50 = models.resnet50(pretrained=True)

        self.encoder = nn.Sequential(
            model_resnet50.conv1,
            model_resnet50.bn1,
            model_resnet50.relu,
            model_resnet50.maxpool,
            model_resnet50.layer1,
            model_resnet50.layer2,
            model_resnet50.layer3,
            model_resnet50.layer4,
            model_resnet50.avgpool,
        )

    def forward(self, input):
        """Forward the LeNet."""
        feat = self.encoder(input)
        feat = feat.view(-1, 2048)
        return feat

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.restored = False

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 31),
        )

    def forward(self, feat):
        out = self.classifier(feat.view(-1, 2048))
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.restored = False

        self.discriminator = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072, 2),
        )

    def forward(self, feat):
        out = self.discriminator(feat)
        return out
