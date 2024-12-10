import torch
import torch.nn as nn
from torchvision import models

class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        print("using VGG16")
        super(CustomVGG16, self).__init__()

        self.vgg16 = models.vgg16(weights='IMAGENET1K_V1')

        self.vgg16.classifier[2] = nn.Dropout(0.5)  # First Dropout with 50% probability
        self.vgg16.classifier[3] = nn.Dropout(0.5)  # First Dropout with 50% probability
        # Dropout after the second hidden layer
        self.vgg16.classifier[6] = nn.Linear(self.vgg16.classifier[6].in_features, num_classes)

        # Add another Dropout after the second fully connected layer
        self.vgg16.classifier[4] = nn.Dropout(0.5)  # Second Dropout with 50% probability

    def forward(self, x):
        x = self.vgg16(x)
        return x
