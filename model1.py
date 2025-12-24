import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
