import torch
import torch.nn as nn

class AudioCNNEnhanced(nn.Module):
    def __init__(self, num_classes):
        print("using AudioCNNEnhanced")
        super(AudioCNNEnhanced, self).__init__()
        self.model = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # Replacing MaxPool with stride-2 Conv
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Second Convolutional Block
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Replacing MaxPool with stride-2 Conv
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Third Convolutional Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Replacing MaxPool with stride-2 Conv
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Global Average Pooling to reduce the number of parameters
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            # Fully Connected Layers
            nn.Linear(64, 32),  # Reduced the number of neurons
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)