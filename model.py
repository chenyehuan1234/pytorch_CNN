import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 5,padding=2)
        self.sig = nn.Sigmoid()
        self.c2=nn.Conv2d(6, 16, 5,padding=2)
