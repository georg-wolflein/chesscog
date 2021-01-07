"""Module containing the CNN architecture definitions of the candidate piece classifiers.
"""

from torch import nn
from torchvision import models
import torch.nn.functional as F
import functools
from recap import CfgNode as CN

from chesscog.core.registry import Registry
from chesscog.core.models import MODELS_REGISTRY

NUM_CLASSES = len({"pawn", "knight", "bishop", "rook", "queen", "king"}) * 2

#: Registry of piece classifiers (registered in the global :attr:`chesscog.core.models.MODELS_REGISTRY` under the key ``PIECE_CLASSIFIER``)
MODEL_REGISTRY = Registry()
MODELS_REGISTRY.register_as("PIECE_CLASSIFIER")(MODEL_REGISTRY)


@MODEL_REGISTRY.register
class CNN100_3Conv_3Pool_3FC(nn.Module):
    """CNN (100, 3, 3, 3) model.
    """

    input_size = 100, 200
    pretrained = False

    def __init__(self):
        super().__init__()
        # Input size: 100x200
        self.conv1 = nn.Conv2d(3, 16, 5)  # 96x196
        self.pool1 = nn.MaxPool2d(2, 2)  # 48x98
        self.conv2 = nn.Conv2d(16, 32, 5)  # 44x94
        self.pool2 = nn.MaxPool2d(2, 2)  # 22x47
        self.conv3 = nn.Conv2d(32, 64, 3)  # 20x45
        self.pool3 = nn.MaxPool2d(2, 2)  # 10x22
        self.fc1 = nn.Linear(64 * 10 * 22, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@MODEL_REGISTRY.register
class CNN100_3Conv_3Pool_2FC(nn.Module):
    """CNN (100, 3, 3, 2) model.
    """

    input_size = 100, 200
    pretrained = False

    def __init__(self):
        super().__init__()
        # Input size: 100x100
        self.conv1 = nn.Conv2d(3, 16, 5)  # 96x196
        self.pool1 = nn.MaxPool2d(2, 2)  # 48x98
        self.conv2 = nn.Conv2d(16, 32, 5)  # 44x94
        self.pool2 = nn.MaxPool2d(2, 2)  # 22x47
        self.conv3 = nn.Conv2d(32, 64, 3)  # 20x45
        self.pool3 = nn.MaxPool2d(2, 2)  # 10x22
        self.fc1 = nn.Linear(64 * 10 * 22, 1000)
        self.fc2 = nn.Linear(1000, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@MODEL_REGISTRY.register
class AlexNet(nn.Module):
    """AlexNet model.
    """

    input_size = 100, 200
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        n = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(n, NUM_CLASSES)
        self.params = {
            "head": list(self.model.classifier[6].parameters())
        }

    def forward(self, x):
        return self.model(x)


@MODEL_REGISTRY.register
class ResNet(nn.Module):
    """ResNet model.
    """

    input_size = 100, 200
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, NUM_CLASSES)
        self.params = {
            "head": list(self.model.fc.parameters())
        }

    def forward(self, x):
        return self.model(x)


@MODEL_REGISTRY.register
class VGG(nn.Module):
    """VGG model.
    """

    input_size = 100, 200
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.vgg11_bn(pretrained=True)
        n = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(n, NUM_CLASSES)
        self.params = {
            "head": list(self.model.classifier[6].parameters())
        }

    def forward(self, x):
        return self.model(x)


@MODEL_REGISTRY.register
class InceptionV3(nn.Module):
    """InceptionV3 model.
    """

    input_size = 299, 299
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(pretrained=True)
        # Auxiliary network
        n = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(n, NUM_CLASSES)
        # Primary network
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, NUM_CLASSES)
        self.params = {
            "head": list(self.model.AuxLogits.fc.parameters()) + list(self.model.fc.parameters())
        }

    def forward(self, x):
        return self.model(x)
