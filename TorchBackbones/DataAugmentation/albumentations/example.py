import os
import sys

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

print(sys.path[1])
proj_root = sys.path[1]


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.masks = sorted(os.listdir(os.path.join(root_dir, 'masks')))
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                     A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                     A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                     ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),  # 随机应用仿射变换：平移，缩放和旋转输入
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.2),  # 随机明亮对比度
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.images[idx])
        mask_path = os.path.join(self.root_dir, 'masks', self.masks[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Convert PIL images to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Apply augmentation
        augmented = self.transform(image=image_np, mask=mask_np)
        image_augmented, mask_augmented = augmented['image'], augmented['mask']

        # Convert back to PIL images
        image_augmented = np.array(image_augmented)
        mask_augmented = np.array(mask_augmented)

        return transforms.ToTensor()(image_augmented), transforms.ToTensor()(mask_augmented)


import matplotlib.pyplot as plt

dataset_root = proj_root + '/Datasets/A_example/seg/train'
# 示例用法
dataset = CustomDataset(root_dir=dataset_root)
for i in range(len(dataset)):
    image, mask = dataset[i]
    image = image.permute(1, 2, 0)
    mask = mask.permute(1, 2, 0)
    # 在这里进行你的操作，例如可视化增强后的图像和掩码
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display Image 1
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].axis('off')

    # Display Image 2
    axes[1].imshow(mask, cmap="bwr")
    axes[1].set_title("mask")
    axes[1].axis('off')

    plt.show()
