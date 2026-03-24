from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
from plot import train_loader


def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True)

    train_data,val_data=Data.random_split(train_data,[int(len(train_data)*0.8),int(len(train_data)*0.2)])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8
                                       )

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8
                                       )

    return train_dataloader,val_dataloader

train_dataloader,val_dataloader=train_val_data_process()


def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()