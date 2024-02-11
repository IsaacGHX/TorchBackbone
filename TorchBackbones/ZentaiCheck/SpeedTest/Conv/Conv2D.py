from ZentaiCheck.devices_confirm import *

import torch
import time

# device = "cpu"
input_size = (1, 32, 32)
k_size1 = 5
k_size2 = 5

# 定义第一个卷积层
conv1 = torch.nn.Conv2d(1, 1, kernel_size=k_size1,bias=False).to(device)

# 定义第二个卷积层
conv2 = torch.nn.Conv2d(1, 1, kernel_size=k_size1,bias=False).to(device)
conv1.train(False)
conv2.train(False)

# 生成随机输入
x = torch.rand(input_size).to(device)

print("Conv Start")
# 测试整个网络前向传播速度
start = time.perf_counter()
for i in range(1024*8):
    y = conv1(x)
    y = conv2(y)
end = time.perf_counter()

print(y)
total = (end - start) / 1024./8.
print("Total time: ", total, "s")
