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
    def __init__(self, annotations_file = None,annotations_df = None,splits = None, labels=None, transform=None, target_transform=None,
                 resize = (224,224)):
        if labels is None:
            labels = ["ABMRh"]
        if splits is None:
            splits = [1]
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        if annotations_df:
            self.img_labels = annotations_df
        else:
            temp = pd.read_csv(annotations_file)
            self.img_labels = temp[temp["split"].isin(splits)]
            
        


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_labels["color"].iloc[idx])
        image =  transforms.Resize((224,224))(image)
        label_values = []
        for label in self.labels:
            label_values.append(self.img_labels[label].iloc[idx])
        if self.transform:
            image = self.transform(image)
        out_labels = [1 if x > 0 else 0 for x in label_values]
        return image, torch.tensor(out_labels)