import torch
import cv2
import numpy as np
from model import Simple3DCNN
from load_video_test import load_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["batting", "bowling", "catching"]
model = Simple3DCNN(num_classes=len(classes))
model.load_state_dict(torch.load("weights/cricket3dcnn.pth", map_location=device))
model.eval()

video_path = r"C:\Users\singh\pratap\cric proj\cricket_dataset\train\batting\v_CricketShot_g01_c02.avi"
video = load_video(video_path)
video_tensor = torch.tensor(video, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(video_tensor)
    _, predicted = torch.max(outputs, 1)
    action = classes[predicted.item()]

print(f"Predicted Action: {action}")
