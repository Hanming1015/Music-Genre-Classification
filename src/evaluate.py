import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.model_VGG16 import CustomVGG16
from src.model_enhanced_CNN import AudioCNNEnhanced
from src.model_resnet import AudioResNet10

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to the same size used in training
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])

# Paths
dataset_path = '../data/music_genre_dataset/mel_spectrograms'

# Create full dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Load dataset
test_dataset = ImageFolder(root='../data/dataset/test', transform=transform)

# Create DataLoaders
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
num_classes = len(os.listdir(dataset_path))  # Number of genre classes
model = CustomVGG16(num_classes=num_classes).to(device)

# Load the saved model weights
model_path = '../models/audio_vgg_1.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, weights_only=False))
    print(f'Model loaded from {model_path}')
else:
    raise FileNotFoundError(f'Model file not found at {model_path}')

# Evaluate the model
model.eval()  # Set the model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():  # No need to calculate gradients during evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Store predictions and labels for metrics calculation
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays for metric calculations
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate overall accuracy
overall_accuracy = 100 * np.mean(all_preds == all_labels)
print(f'\nOverall Test Accuracy: {overall_accuracy:.2f}%')

# Get class names from the original dataset
class_names = full_dataset.classes

# Calculate precision, recall, and F1-score
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

# Extract precision, recall, and F1-score for each class
precision = [report[label]['precision'] for label in class_names]
recall = [report[label]['recall'] for label in class_names]
f1_score = [report[label]['f1-score'] for label in class_names]

# Set up the figure for metrics visualization
plt.figure(figsize=(14, 8))
x = np.arange(len(class_names))

# Plot Precision, Recall, F1-score
width = 0.25
plt.bar(x - width, precision, width=width, label='Precision', color='#6a5acd')  # SlateBlue
plt.bar(x, recall, width=width, label='Recall', color='#ffa07a')  # LightSalmon
plt.bar(x + width, f1_score, width=width, label='F1-score', color='#8fbc8f')  # DarkSeaGreen

# Formatting the plot
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1-score per Class')
plt.xticks(ticks=x, labels=class_names, rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', linestyle='--')

# Save the metrics visualization
plt.savefig('classification_metrics_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and display the confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
# Save the confusion matrix
plt.savefig('confusion_matrix_res.png', dpi=300, bbox_inches='tight')
plt.show()




