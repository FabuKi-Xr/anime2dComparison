from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

class LoadDataset(Dataset):
    def __init__(self, low_quality_folder, high_quality_folder, transform=None):
        self.low_quality_folder = low_quality_folder
        self.high_quality_folder = high_quality_folder
        self.transform = transform

        self.low_quality_images = os.listdir(low_quality_folder)
        self.high_quality_images = os.listdir(high_quality_folder)

    def __len__(self):
        return len(self.low_quality_images)

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_quality_folder, self.low_quality_images[idx])
        high_img_path = os.path.join(self.high_quality_folder, self.high_quality_images[idx])

        low_img = Image.open(low_img_path).convert('RGB')
        high_img = Image.open(high_img_path).convert('RGB')

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img