import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


class Combine_model(nn.Module):
    def __init__(self,model_A,model_B):
        super(Combine_model, self).__init__()
        self.model_a = model_A
        self.model_b = model_B

    def forward(self, x):
        x = self.model_a(x)
        x = self.model_b(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features