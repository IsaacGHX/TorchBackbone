import torch
import random
from torchvision import transforms
import numpy as np

class Mixup(object):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, sample):
        image, label = sample

        # 随机选择另一张图像和标签
        index = random.randint(0, len(image) - 1)
        mix_image = image[index]
        mix_label = label[index]

        # 生成Mixup后的图像和标签
        mix_image = self.alpha * image + (1 - self.alpha) * mix_image
        mix_label = self.alpha * label + (1 - self.alpha) * mix_label

        return mix_image, mix_label


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # 创建一张全零、全一和全0.2的图像
    zero_image = torch.zeros(3, 224, 224)
    ones_image = torch.ones(3, 224, 224)
    image = torch.cat([zero_image.unsqueeze(0), ones_image.unsqueeze(0)], dim=0)
    print(image.shape)
    label = torch.tensor([1.0, 0.0])

    # 应用转换
    transformed_image, transformed_label = Mixup(alpha=0.2)((image, label))
    print(transformed_image.shape)
    # 创建一行三列的展示区域
    plt.figure(figsize=(12, 4))

    # 显示全零图
    plt.subplot(1, 3, 1)
    plt.imshow(zero_image.permute(1, 2, 0))
    plt.title('All Zeros')
    plt.axis('off')

    # 显示全一图
    plt.subplot(1, 3, 2)
    plt.imshow(ones_image.permute(1, 2, 0))
    plt.title('All Ones')
    plt.axis('off')

    # 显示全0.2图
    plt.subplot(1, 3, 3)
    plt.imshow(transformed_image[0].squeeze(0).permute(1, 2, 0))
    plt.title('Mixed up')
    plt.axis('off')

    plt.show()
