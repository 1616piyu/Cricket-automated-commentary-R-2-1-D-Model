import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CricketVideoDataset(Dataset):
    """
    Loads cricket videos and returns tensor shape:
        (C, T, H, W) = (3, max_frames, 112, 112)
    """

    def __init__(self, root_dir, max_frames=32, transform=None):
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))
        self.samples = []

        # Scan all class folders & collect video paths
        for label_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)

            for video_name in os.listdir(class_folder):
                video_path = os.path.join(class_folder, video_name)
                self.samples.append((video_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def read_video(self, path):
        """Loads frames, resizes to 112×112 and uniformly samples max_frames."""
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

        # If somehow empty, return black frames
        if len(frames) == 0:
            return np.zeros((self.max_frames, 112, 112, 3), dtype=np.uint8)

        # Uniform temporal sampling
        indices = np.linspace(0, len(frames) - 1, self.max_frames).astype(int)
        frames = [frames[i] for i in indices]

        return np.array(frames)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frames = self.read_video(video_path)
        frames = frames.astype(np.float32) / 255.0  # normalize

        # Convert from (T,H,W,C) → (C,T,H,W)
        frames = np.transpose(frames, (3, 0, 1, 2))

        tensor_frames = torch.tensor(frames, dtype=torch.float32)
        tensor_label = torch.tensor(label, dtype=torch.long)

        return tensor_frames, tensor_label
