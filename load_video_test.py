import torch
import torchvision
from torchvision.io import read_video
from torchvision.transforms import Resize
from PIL import Image
import numpy as np

def load_video(path, max_frames=64, resize=(112, 112)):
    """
    Loads a video, limits frames, resizes, and converts to float32 tensor.
    
    Args:
        path (str): Path to the video file
        max_frames (int): Maximum number of frames to load
        resize (tuple): Desired (height, width)
    
    Returns:
        video (torch.Tensor): Shape (C, T, H, W), dtype=float32
    """
    try:
        # Read video using torchvision
        video, _, _ = read_video(path, pts_unit="sec")  # video shape: (T, H, W, C)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

    # Limit frames
    if video.shape[0] > max_frames:
        video = video[:max_frames]
    elif video.shape[0] < max_frames:
        # Pad frames if video is shorter than max_frames
        pad_frames = max_frames - video.shape[0]
        pad_tensor = torch.zeros((pad_frames, *video.shape[1:]), dtype=video.dtype)
        video = torch.cat((video, pad_tensor), dim=0)

    # Resize frames
    resized_frames = []
    for frame in video:
        img = Image.fromarray(frame.numpy())
        img = img.resize(resize)
        frame_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # (C, H, W)
        resized_frames.append(frame_tensor)

    video = torch.stack(resized_frames, dim=1)  # (C, T, H, W)

    # Convert to float32 and normalize
    video = video.float() / 255.0

    return video  # shape: (C, T, H, W)











