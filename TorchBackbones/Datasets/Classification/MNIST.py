import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

# 参数
batch_size = 32
val_split = 0.86
num_classes = 10
augment = True  # 是否数据增强

# 数据增强
if augment:
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize
    ])
else:
    transform = transforms.ToTensor()

# 下载并加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='D:/datasets',
                                           train=True,
                                           transform=transform,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='D:/datasets',
                                          train=False,
                                          transform=transforms.ToTensor())

print("MNIST data pack downloaded! www")

# 划分训练验证集
train_size = int(len(train_dataset) * val_split)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 数据标准化
train_dataset.transform = transforms.Compose([transforms.ToTensor(), normalize])
val_dataset.transform = transforms.Compose([transforms.ToTensor(), normalize])
test_dataset.transform = transforms.Compose([transforms.ToTensor(), normalize])

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("train length: ", len(train_loader) * batch_size)
print("val length: ", len(val_loader) * batch_size)
print("test length: ", len(test_loader) * batch_size)
# 取样本索引0
imgs, targets = next(iter(train_loader))
img, target = imgs[0], targets[0]
target = torch.tensor([target])  # 将整数转换为张量

print("single img shape: ", img.shape)
print("target e.g. is:", target)
print("basic augmentation: ", augment)

__all__ = ['train_loader', 'val_loader', 'test_loader', 'batch_size', 'val_split', 'num_classes']

if __name__ == '__main__':
    from Visualiazation.ShowDatasets import show_square_pics

    show_square_pics(train_dataset, k=4, color_mode='gray')
