import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        # super 解决了多重继承中，调用父类方法中，解决的一系列问题
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        # 池化层只会改变特征矩阵的高和宽
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)               # output(16, 14, 14)
        x = F.relu(self.conv2(x))       # input(16, 14, 14) output(32, 10, 10)
        x = self.pool2(x)               # output(32, 5, 5)
        x = x.view(-1, 32*5*5)          # output(32*5*5)
        x = F.relu(self.fc1(x))         # output(120)
        x = F.relu(self.fc2(x))         # output(84)
        x = self.fc3(x)                 # output(10)
        return x

# debug

# import torch
# input1 = torch.rand([32, 3, 32, 32])
# model = LeNet()
# print(model)
# output = model(input1)