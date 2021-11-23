import torch
from torch import nn
from typing import List, Optional

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class Bottleneck(nn.Module):
    # ResNet bottleneck block, 1x1conv-3x3conv-1x1conv.
    # downsample between stage layers to halve the size of input,
    # cooperated with 3x3conv stride=2

    # Normally the 3td conv possesses 4x channels compared to previous ones
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()
        # Notice: In the first bottleneck of every residual stage,
        #         set 3x3conv stride=2 to halve the feature_map size,
        #         except the first residual stage (also named stage 2),
        #         which has been halved by MaxPooling with stirde=2 in stage 1.
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample == None:
            identity = x
        else:
            identity = self.downsample(x)

        out += identity  # pixel-wise addtition
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    # Generic building func for ResNet-n
    def __init__(self, layers: List[int], num_classess=2):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.bottleneck = Bottleneck
        # The followling layers define stage 1(befor residual blocks)
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # The following layers define stage 2-5(residual blocks)
        self.layer1 = self._make_layer(16, layers[0])
        self.layer2 = self._make_layer(32, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)
        self.layer4 = self._make_layer(128, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * self.bottleneck.expansion, num_classess)

    # Build residual block
    def _make_layer(self, channels, num_bottleneck, stride=1):
        # 'stride' for 3x3conv or 1x1conv in downsample
        # channel=64 in first residual stage
        # stride=2 3x3conv & 1x1conv(downsmaple) in bottlenect 1 in in stage 2-5
        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, channels * self.bottleneck.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channels * self.bottleneck.expansion))
        layers = []
        # Append the first bottleneck of residual block
        layers.append(self.bottleneck(
            self.in_channels, channels, stride, downsample))
        # Append the rest bottlenecks of residual block
        self.in_channels *= self.bottleneck.expansion
        # For stage 3-5, in_channels is half of the out_channels of previous stage
        if channels != 16:  # Indicate stage 3-5
            self.in_channels = int(self.in_channels / 2)
        for _ in range(1, num_bottleneck):
            layers.append(self.bottleneck(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stage 1
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # Stage 2-5
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # GlobalAvgPool-FC
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


# Construct ResNet-n
def resnet50(num_classes):
    return ResNet([3, 4, 6, 3], num_classes)
