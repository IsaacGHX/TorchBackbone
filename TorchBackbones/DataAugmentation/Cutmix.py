import torch
import random
import numpy as np

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, data):
        image, target = data
        lam = np.random.beta(self.alpha, self.alpha)

        image_size = image.size()
        patch_size = (int(image_size[1] * np.sqrt(1 - lam)), int(image_size[2] * np.sqrt(1 - lam)))
        cx, cy = np.random.randint(image_size[1]), np.random.randint(image_size[2])
        x1 = max(0, cx - patch_size[0] // 2)
        x2 = min(image_size[1], cx + patch_size[0] // 2)
        y1 = max(0, cy - patch_size[1] // 2)
        y2 = min(image_size[2], cy + patch_size[1] // 2)

        mixed_image = image.clone()
        mixed_image[:, x1:x2, y1:y2] = image[:, x1:x2, y1:y2]

        lam = 1 - ((x2 - x1) * (y2 - y1) / (image_size[1] * image_size[2]))
        target = lam * target + (1.0 - lam) * target

        return mixed_image, target

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # 创建一张全零、全一和全0.2的图像
    zero_image = torch.zeros(3, 224, 224)
    ones_image = torch.ones(3, 224, 224)
    image = torch.cat([zero_image.unsqueeze(0), ones_image.unsqueeze(0)], dim=0)
    print(image.shape)
    label = torch.tensor([1.0, 0.0])

    # 应用转换
    transformed_image, transformed_label = CutMix(alpha=0.2)((image, label))
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
