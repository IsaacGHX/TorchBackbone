"""
这是一个迁移性测试，每个不同的数据集需要重新改进去调整
"""

from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms


def test_num_workers(trainset, num_workers):
    train_loader = torch.utils.data.DataLoader(
        trainset, shuffle=True, num_workers=num_workers, batch_size=64, pin_memory=True
    )
    start = time()
    for epoch in range(1, 2):
        for i, data in enumerate(train_loader):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
    return end - start


if __name__ == "__main__":
    print(f"num of all CPU cores: {mp.cpu_count()}")
    all_time_list = []
    for num_workers in range(2, mp.cpu_count() // 2, 2):
        cost = test_num_workers(torchvision.datasets.MNIST(root="D:/datasets/", train=True,
                                                           download=True,
                                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                                         transforms.Normalize((0.1307,),
                                                                                                              (
                                                                                                                  0.3081,))])),
                                num_workers)
        all_time_list.append(cost)
    import numpy as np

    all_time_list = np.array(all_time_list)
    print(2 * (1 + np.argmin(all_time_list)), "num_works is the fastest.")
