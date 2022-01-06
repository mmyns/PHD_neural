import os
from PIL import Image
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, frames,splits = None, labels=None, transform=None, target_transform=None,
                 resize = (224,224)):
        if labels is None:
            labels = ["ABMRh"]
        if splits is None:
            splits = [1]
        self.labels = labels
        temp = pd.read_csv(annotations_file)
        self.img_labels = temp[temp["split"].isin(splits)]
        self.transform = transform
        self.target_transform = target_transform
        self.frames = frames
        self.resize = resize


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        images = []
        for frame in self.frames:
            img_path = self.img_labels[str(frame)].iloc[idx]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img,(448,448))
            cvuint8 = (img/256).astype(np.uint8)
            toadd = torch.tensor(cvuint8)
            toadd = toadd[np.newaxis, :]
            toadd = toadd.repeat(3, 1, 1)
            images.append(toadd)
        image = torch.cat(images,0)
        label_values = []
        for label in self.labels:
            label_values.append(self.img_labels[label].iloc[idx])
        if self.transform:
            image = self.transform(image)
        out_labels = [1 if x > 0 else 0 for x in label_values]
        return image, torch.tensor(out_labels)