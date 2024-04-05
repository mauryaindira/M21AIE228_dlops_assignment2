import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, auc


# Load USPS dataset
usps = fetch_openml('usps', version=2)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load USPS dataset
usps_train = datasets.USPS(root='./data', train=True, download=True, transform=transform)
usps_test = datasets.USPS(root='./data', train=False, download=True, transform=transform)



# Split dataset into training and validation sets
usps_train, usps_val = random_split(usps_train, [6000, 1291])


# Define dataloaders
train_loader = DataLoader(usps_train, batch_size=64, shuffle=True)
val_loader = DataLoader(usps_val, batch_size=64, shuffle=False)
test_loader = DataLoader(usps_test, batch_size=64, shuffle=False)

# Define CNN architecture # Basic configurations
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*16*2, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Configuration 2: CNN with Dropout
class CNNWithDropout(nn.Module):
    def __init__(self):
        super(CNNWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Configuration 3: CNN with Batch Normalization
class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super(CNNWithBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize models
cnn_model = CNN().to(device)
cnn_model_1 = CNNWithDropout().to(device)
cnn_model_2 = CNNWithBatchNorm().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
cnn_optimizer_1 = optim.Adam(cnn_model_1.parameters(), lr=0.001)
cnn_optimizer_1 = optim.Adam(cnn_model_2.parameters(), lr=0.001)
# TensorBoard SummaryWriter
writer = SummaryWriter()

# TensorBoard SummaryWriter
writer = SummaryWriter()
# Training function
def train(model, optimizer, criterion, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            if i % 100 == 0:  # Log every 100 batches
                writer.add_scalar('training_loss', loss.item(), epoch * len(train_loader) + i)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item() * images.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        # Log validation loss
        writer.add_scalar('validation_loss', val_loss, epoch)

        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Training Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')


# Train CNN
print("\nTraining CNN...")
train(cnn_model, cnn_optimizer, criterion, train_loader, val_loader)
train(cnn_model_1, cnn_optimizer_1, criterion, train_loader, val_loader)
train(cnn_model_2, cnn_optimizer_2, criterion, train_loader, val_loader)

# Test  CNN
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += lbels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')


# Test CNN
print("\nTesting CNN...")
test(cnn_model, test_loader)
test(cnn_model_1, test_loader)
test(cnn_model_2, test_loader)


