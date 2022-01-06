import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3,stride = 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5,stride = 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 5,stride = 2)
        self.conv4 = torch.nn.Conv2d(64, 64, 5,stride = 2)
        self.conv4 = torch.nn.Conv2d(64, 64, 5,stride = 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(59904, 32)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x,3)
        
        x = F.relu(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.5, inplace=False)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.5, inplace=False)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features