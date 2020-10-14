from torch import nn
import torch.nn.functional as F


class CNN100(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: 100x100
        self.conv1 = nn.Conv2d(3, 16, 5)  # 96
        self.pool1 = nn.MaxPool2d(2, 2)  # 48
        self.conv2 = nn.Conv2d(16, 32, 5)  # 44
        self.pool2 = nn.MaxPool2d(2, 2)  # 22
        self.conv3 = nn.Conv2d(32, 64, 3)  # 20
        self.pool3 = nn.MaxPool2d(2, 2)  # 10
        self.fc1 = nn.Linear(64 * 10 * 10, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN50(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: 50x50
        self.conv1 = nn.Conv2d(3, 16, 3)  # 48
        self.pool1 = nn.MaxPool2d(2, 2)  # 24
        self.conv2 = nn.Conv2d(16, 32, 3)  # 22
        self.pool2 = nn.MaxPool2d(2, 2)  # 11
        self.fc1 = nn.Linear(32 * 11 * 11, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
