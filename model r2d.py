import torch
import torch.nn as nn


class R2Plus1DBlock(nn.Module):
    """
    Basic (2+1)D block:
    - Spatial conv (1x3x3)
    - Temporal conv (3x1x1)
    with a residual skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # Mid channels (official R(2+1)D decomposition)
        mid = int((3 * 3 * 3 * in_channels * out_channels) /
                  (3 * 3 * in_channels + 3 * out_channels))

        # Spatial conv
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True)
        )

        # Temporal conv
        self.temporal = nn.Sequential(
            nn.Conv3d(mid, out_channels, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.spatial(x)
        out = self.temporal(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class R2Plus1D(nn.Module):
    """
    R(2+1)D model (ResNet-18 style)
    Input: (B, 3, T, 112, 112)
    """
    def __init__(self, num_classes=3, layers=(2, 2, 2, 2)):
        super().__init__()
        self.in_channels = 64

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3),
                         stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # Residual layers
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2],
