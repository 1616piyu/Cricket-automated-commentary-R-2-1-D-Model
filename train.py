import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CricketVideoDataset
from load_video_test import load_video
from model import Simple3DCNN  # Your 3D CNN model

# -------------------------
# Paths and Parameters
# -------------------------
train_dir = r"C:\Users\singh\pratap\cric proj\cricket_dataset\train"
val_dir = r"C:\Users\singh\pratap\cric proj\cricket_dataset\val"
num_classes = 3  # batting, bowling, catching
frames_per_clip = 64
batch_size = 2
num_epochs = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Dataset & DataLoader
# -------------------------
train_dataset = CricketVideoDataset(train_dir, max_frames=frames_per_clip)
val_dataset = CricketVideoDataset(val_dir, max_frames=frames_per_clip)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# Model, Loss, Optimizer
# -------------------------
model = Simple3DCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in train_loader:
        # videos: (B, T, C, H, W)
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

# -------------------------
# Save Model
# -------------------------
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/cricket3dcnn.pth")
print("✅ Model saved at weights/cricket3dcnn.pth")

