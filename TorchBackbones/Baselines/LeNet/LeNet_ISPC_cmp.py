from Datasets.Classification.MNIST import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


if __name__ == '__main__':
    from ZentaiCheck.devices_confirm import *

    print(torch.cuda.current_device())  # 如果没有使用GPU则打印-1

    model = LeNet5().to(device)
    loss = nn.CrossEntropyLoss().to(device)

    from torch.optim import SGD

    from Train_Paradigm.VeryNormal import train

    import time

    start = time.time()
    train(model, train_loader, val_loader, loss, SGD, batch_size, device=device, lr=1e-1,
          epochs=1, summarize=True, input_size=(batch_size, 1, 32, 32), save=False)
    end = time.time()
    print(f"Total time is:", end - start)
