from asyncore import write
from itertools import count
from platform import release
from pyexpat import features, model
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from yaml import load


class Net(nn.Module):
    def __init__(self, n_channnel, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channnel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flattern()
        # self.l1 = nn.Linear()

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu
        )

        def forward(self, x):
            x1 = self.features(x)
            x2 = self.flatten(x1)
            return x2
