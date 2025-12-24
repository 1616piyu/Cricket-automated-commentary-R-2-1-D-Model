# test_dataset.py
from dataset import CricketVideoDataset
from torch.utils.data import DataLoader

dataset = CricketVideoDataset(r"C:\Users\singh\pratap\cric proj\cricket_dataset\train")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for videos, labels in dataloader:
    print("Batch video shape:", videos.shape)
    print("Labels:", labels)
    break



