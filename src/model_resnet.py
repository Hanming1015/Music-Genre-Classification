import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AudioResNet10(nn.Module):
    def __init__(self, num_classes):
        print("using AudioResNet10")
        super(AudioResNet10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=7, stride=2, padding=3, bias=False)  # Further reduced number of channels from 8 to 6
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(6, 6, blocks=1, stride=1)  # Further reduced number of channels
        self.layer2 = self._make_layer(6, 12, blocks=1, stride=2)
        self.layer3 = self._make_layer(12, 24, blocks=1, stride=2)
        self.layer4 = self._make_layer(24, 48, blocks=1, stride=2)  # Reduced final layer channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48, num_classes)  # Reduced number of channels from 64 to 48

        # Further reduce Dropout layers before fully connected layers
        self.dropout1 = nn.Dropout(0.3)  # Reduced Dropout after layer2
        self.dropout2 = nn.Dropout(0.3)  # Reduced Dropout after layer3
        self.dropout3 = nn.Dropout(0.3)  # Reduced Dropout before the fully connected layer

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout1(x)  # Apply dropout after layer2

        x = self.layer3(x)
        x = self.dropout2(x)  # Apply dropout after layer3
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Apply Dropout before the fully connected layer
        x = self.dropout3(x)
        x = self.fc(x)

        return x
