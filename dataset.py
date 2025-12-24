import os
import torch
from torch.utils.data import Dataset
from load_video_test import load_video


class CricketVideoDataset(Dataset):
    def __init__(self, root_dir, max_frames=64):
        """
        root_dir structure:
        root_dir/
            batting/
            bowling/
        """
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.video_paths = []
        self.labels = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith(".mp4"):
                    self.video_paths.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video_tensor = load_video(video_path, self.max_frames)
        return video_tensor, torch.tensor(label)




