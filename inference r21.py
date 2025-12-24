import torch
import numpy as np
from model import R2Plus1D
from load_video_test import load_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Action classes
classes = ["batting", "bowling", "catching"]

# Load model
model = R2Plus1D(num_classes=len(classes))
model.load_state_dict(torch.load("weights/cricket_r2plus1d.pth", map_location=device))
model = model.to(device)
model.eval()

# Path to test video
video_path = r"C:\Users\singh\pratap\cric proj\cricket_dataset\train\batting\v_CricketShot_g01_c02.avi"

# Load video -> returns (C,T,H,W)
video = load_video(video_path)

# Convert to tensor (1,C,T,H,W)
video_tensor = torch.tensor(video, dtype=torch.float32).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    outputs = model(video_tensor)
    _, predicted = torch.max(outputs, 1)
    action = classes[predicted.item()]

print(f"\nPredicted Action: {action}\n")
