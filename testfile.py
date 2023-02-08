import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Alex_net(nn.Module):
    def __init__(self, hidden_dim, alpha, device):
        super().__init__()

        self.alex_net = nn.Sequential(
            # 这⾥，我们使⽤⼀个11*11的更⼤窗⼝来捕捉对象。
            # 同时，步幅为4，以减少输出的⾼度和宽度。
            # 另外，输出通道的数⽬远⼤于LeNet
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
            nn.Conv2d(48, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
            # 除了最后的卷积层，输出通道的数量进⼀步增加。
            # 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
            nn.Conv2d(128, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 2048), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 128)).to(device)
        self.fc1 = nn.Linear(128, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, 256).to(device)
        self.fc3 = nn.Linear(256, 1).to(device)
        self.relu = nn.ReLU()
        self.alpha = alpha
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.alpha)
    def forward(self, state):
        x = self.relu(self.alex_net(state))
        x = self.relu(self.fc2(x))
        x = self.relu(x)
        return x


x = np.zeros((224, 224, 3))
x = np.reshape(x, (3, 224, 224))
x = torch.tensor(np.array(x),dtype=torch.float)
x = torch.unsqueeze(x, dim=0).to(device)  # 在第一维度添加维度
net = Alex_net(128, 0.001, device)
temp = net.forward(x)
for layer in net.alex_net:
    x=layer(x)
print(layer.__class__.__name__,'output shape:\t',x.shape)

