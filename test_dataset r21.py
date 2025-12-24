from dataset import CricketVideoDataset
from model import R2Plus1D
import torch

# Path to validation dataset
val_dir = r"C:\Users\singh\pratap\cric proj\cricket_dataset\val"

# Class names
classes = ["batting", "bowling", "catching"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load R(2+1)D model
model = R2Plus1D(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("weights/cricket_r2plus1d.pth", map_location=device))
model.eval()

# Load validation dataset
dataset = CricketVideoDataset(val_dir)

correct, total = 0, 0

# Evaluation loop
with torch.no_grad():
    for video, label in dataset:
        video = video.unsqueeze(0).to(device)   # (1, C, T, H, W)
        label = label.to(device)

        output = model(video)
        _, pred = torch.max(output, 1)

        if pred.item() == label.item():
            correct += 1
        total += 1

accuracy = (correct / total) * 100
print(f"Validation Accuracy: {accuracy:.2f}%")
