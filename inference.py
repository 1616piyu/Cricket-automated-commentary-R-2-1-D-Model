import torch
import os
from load_video_test import load_video
from model import Simple3DCNN  # make sure your model class name matches
import torch.nn as nn

# ✅ Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "weights/cricket3dcnn.pth"

# Classes
classes = ["batting", "bowling", "catching"]

# Load model
model = Simple3DCNN(num_classes=len(classes))  # ensure num_classes=3
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# Test video path (replace with your video)
video_path = r"C:\Users\singh\pratap\cric proj\cricket_dataset\train\batting\bat2.mp4"

# Load video
video_tensor = load_video(video_path)           # (T, C, H, W)
video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, C, T, H, W)
print(f"Video shape: {video_tensor.shape}, Data type: {video_tensor.dtype}")

# Forward pass
with torch.no_grad():
    outputs = model(video_tensor)
    _, predicted = torch.max(outputs, 1)

# Display result
print(f"Predicted action: {classes[predicted.item()]}")
print("Raw model outputs:", outputs.cpu().numpy())
print("Predicted index:", predicted.item())
print("Expected class order:", classes)








