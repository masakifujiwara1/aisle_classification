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
import torch.nn.functional as F
from yaml import load

# input 64

INPUT = 64  # 64
BATCH_SIZE = 16  # 16
MAX_DATA = 10000

INPUT_ = int(INPUT/4)


class Net(nn.Module):
    def __init__(self, n_channnel, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channnel, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # self.l1 = nn.Linear()
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)

        self.classifier = nn.Linear(32*(INPUT_)**2, 3)
        torch.nn.init.kaiming_normal_(self.classifier.weight)

        # self.softmax = F.softmax()

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.maxpool,
            self.conv2,
            self.relu,
            self.maxpool,
            self.conv3,
            self.relu,
            # self.maxpool,
            # self.conv4,
            # self.relu
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        # x2 = x1.view(x1.size(0), -1)
        x3 = self.classifier(x2)
        # x4 = F.softmax(x3, dim=1)

        # debug
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        return x3
        # return x4


class deep_learning:
    def __init__(self, n_channel=3, n_out=3):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_out).to(self.device)
        print(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.net.parameters(), eps=1e-2, lr=0.001, weight_decay=5e-4)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        self.first_flag = True
        self.count = 0

    def detect_and_trains(self, img, aisle_class):
        self.net.train()

        if self.first_flag:
            self.x_cat = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.t_cat = torch.tensor(
                [aisle_class], dtype=torch.float32, device=self.device).unsqueeze(0)
            self.first_flag = False
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        t = torch.tensor([aisle_class], dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        self.x_cat = torch.cat([self.x_cat, x], dim=0)
        self.t_cat = torch.cat([self.t_cat, t], dim=0)

        dataset = TensorDataset(self.x_cat, self.t_cat)
        train_dataset = DataLoader(
            dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=True)

        for x_train, t_train in train_dataset:
            x_train.to(self.device, non_blocking=True)
            t_train.to(self.device, non_blocking=True)
            break

        self.optimizer.zero_grad()
        y_train = self.net(x_train)
        # print(y_train.shape)
        # print(t_train.shape)
        t_train = t_train.view(t_train.size(0), -1)
        loss = self.criterion(y_train, t_train)
        loss.backward()
        # pred = torch.argmax(y_train, dim=1)
        # print(pred)
        self.optimizer.step()

        self.count += 1

        action_value_training = self.net(x)
        # print(action_value_training)
        # print(torch.argmax(action_value_training))

        if self.x_cat.size()[0] > MAX_DATA:
            self.x_cat = torch.empty(1, 3, 128, 128).to(self.device)
            self.t_cat = torch.empty(1, 3).to(self.device)
            self.first_flag = True
            print("reset dataset")

        return action_value_training[0][0].item(), action_value_training[0][1].item(), action_value_training[0][2].item(), loss.item()

    def detect(self, img):
        self.net.eval()
        x_test_ten = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0, 3, 1, 2)
        action_value_test = self.net(x_test_ten)
        return action_value_test.tolist()

    def test(self):
        input = torch.randn(1, 3, 128, 128).to(self.device)
        self.net.forward(input)
        # print(self.net.x1.shape)
        # print(self.net.x2.shape)
        # print(self.net.x3.shape)

    def save(self, save_path):
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(self.net.state_dict(), path + '/model_gpu.pt')

    def load(self, load_path):
        self.net.load_state_dict(torch.load(load_path))


if __name__ == '__main__':
    dl = deep_learning()
    dl.test()
