import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CricketVideoDataset
from model import R2Plus1D

# Paths
train_dir = r"C:\Users\singh\pratap\cric proj\cricket_dataset\train"
val_dir   = r"C:\Users\singh\pratap\cric proj\cricket_dataset\val"

# Hyperparameters
num_classes = 3
frames_per_clip = 32
batch_size = 2
num_epochs = 10
learning_rate = 1e-4

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets & Loaders
train_dataset = CricketVideoDataset(train_dir, max_frames=frames_per_clip)
val_dataset   = CricketVideoDataset(val_dir,   max_frames=frames_per_clip)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# Model
model = R2Plus1D(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # Validation
    model.eval()
