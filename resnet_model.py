
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import timm


class NeuralNetwork(nn.Module):
    def __init__(self,final_classes = 2):
        super(NeuralNetwork, self).__init__()
        model_resnet = timm.create_model('resnet18', pretrained=True)
        model_resnet.fc = torch.nn.Linear(512, 512)
        for param in model_resnet.parameters():
            param.requires_grad = False 
        for param in model_resnet.fc.parameters():
            param.requires_grad = True
        self.startmodel = model_resnet
        self.final_layer = torch.nn.Linear(512, final_classes)

    def forward(self, x):
        x = F.dropout(F.relu(self.startmodel(x)),0.5)
        x = self.final_layer(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def get_model_name(self):
        return "Resnet_model_Tensorflow"