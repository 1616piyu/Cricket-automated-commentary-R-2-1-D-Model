from dataset import CricketVideoDataset
from model import Simple3DCNN
import torch

val_dir = r"C:\Users\singh\pratap\cric proj\cricket_dataset\val"
classes = ["batting", "bowling", "catching"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Simple3DCNN(num_classes=len(classes))
model.load_state_dict(torch.load("weights/cricket3dcnn.pth", map_location=device))
model.eval()

dataset = CricketVideoDataset(val_dir)
correct, total = 0, 0

with torch.no_grad():
    for video, label in dataset:
        video = video.unsqueeze(0).to(device)
        output = model(video)
        _, pred = torch.max(output, 1)
        if pred.item() == label.item():
            correct += 1
        total += 1

print(f"Validation Accuracy: {(correct / total) * 100:.2f}%")
