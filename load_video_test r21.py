import cv2
import numpy as np

def load_video(path, max_frames=32):
    """
    Loads a video, resizes frames to 112x112, converts to RGB,
    uniformly samples frames, and returns tensor shape:
        (C, T, H, W)
    """

    frames = []
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing
        frame = cv2.resize(frame, (112, 112))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # If video unreadable → return black clip
    if len(frames) == 0:
        return np.zeros((3, max_frames, 112, 112), dtype=np.float32)

    # Uniform sampling
    indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    frames = [frames[i] for i in indices]

    # Convert to numpy
    frames = np.array(frames, dtype=np.float32) / 255.0

    # Reorder to (C, T, H, W)
    frames = np.transpose(frames, (3, 0, 1, 2))

    return frames
