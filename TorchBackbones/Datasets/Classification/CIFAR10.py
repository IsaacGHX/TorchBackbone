import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.nn.functional import one_hot

# 参数
batch_size = 32
val_split = 0.86
num_classes = 10
augment = True  # 是否数据增强

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10(root='D:/datasets',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='D:/datasets',
                                            train=False,
                                            transform=transforms.ToTensor())

# print(train_dataset[0])

# 划分训练验证集
train_size = int(len(train_dataset) * val_split)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 数据标准化
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_dataset.transform = transforms.Compose([transforms.ToTensor(), normalize])
val_dataset.transform = transforms.Compose([transforms.ToTensor(), normalize])
test_dataset.transform = transforms.Compose([transforms.ToTensor(), normalize])

# 数据增强
if augment:
    train_dataset.transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((32, 32), interpolation=3, antialias=True),
        transforms.ToTensor(),
        normalize
    ])

__all__ = ['train_dataset', 'val_dataset', 'test_dataset', 'batch_size', 'val_split', 'num_classes']

if __name__ == '__main__':
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("train length: ", len(train_loader) * batch_size)
    print("val length: ", len(val_loader) * batch_size)
    print("test length: ", len(test_loader) * batch_size)
    # 取样本索引0
    imgs, targets = next(iter(train_loader))

    # targets = one_hot(targets)
    img, target = imgs[0], targets[0]
    target = torch.tensor([target])  # 将整数转换为张量

    print("single img shape: ", img.shape)
    print("target e.g. is:", target)
    print("basic augmentation: ", augment)

    from Visualiazation.ShowDatasets import show_square_pics

    show_square_pics(train_dataset, k=4, color_mode='color')
