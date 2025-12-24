import cv2
import numpy as np

def load_video(path, max_frames=32):
    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return np.zeros((max_frames, 112, 112, 3), dtype=np.uint8)

    indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    frames = [frames[i] for i in indices]
    frames = np.array(frames) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
    return frames
