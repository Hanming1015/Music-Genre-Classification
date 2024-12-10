import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

from model_enhanced_CNN import AudioCNNEnhanced
from src.model_resnet import AudioResNet10
from model_VGG16 import CustomVGG16

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transformations with augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Paths
dataset_path = '../data/music_genre_dataset/mel_spectrograms'

# Load the datasets
train_dataset = ImageFolder(root='../data/dataset/train', transform=transform)
val_dataset = ImageFolder(root='../data/dataset/val', transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize the model
num_classes = len(os.listdir(dataset_path))
model = AudioResNet10(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.0005)
# optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

# Lists to store the loss and accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Training phase
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Calculate average training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Evaluation phase
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average val loss and accuracy
    val_loss = running_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    # Print epoch information
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Step the scheduler
    scheduler.step(val_loss)

# Save the model
os.makedirs('../models', exist_ok=True)
torch.save(model.state_dict(), '../models/audio_res_1.pth')

# Plotting the results
plt.figure(figsize=(15, 10))

# # Loss Plot
# plt.subplot(2, 2, 1)
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# Accuracy Plot
plt.subplot(2, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

# Save the entire figure
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')

# # Loss Difference Plot
# plt.subplot(2, 2, 3)
# loss_difference = [train - test for train, test in zip(train_losses, val_losses)]
# plt.plot(loss_difference, label='Loss Difference (Train - Validation)')
# plt.title('Loss Difference vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss Difference')
# plt.legend()
#
# # Accuracy Difference Plot
# plt.subplot(2, 2, 4)
# accuracy_difference = [train - test for train, test in zip(train_accuracies, val_accuracies)]
# plt.plot(accuracy_difference, label='Accuracy Difference (Train - Validation)')
# plt.title('Accuracy Difference vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy Difference (%)')
# plt.legend()

plt.tight_layout()
plt.show()
