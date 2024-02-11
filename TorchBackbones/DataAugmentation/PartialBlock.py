import torch
from torchvision import transforms


class PartialBlockMask(object):
    def __init__(self, block_size=(16, 16)):
        self.block_size = block_size

    def __call__(self, img):
        h, w = img.size(1), img.size(2)  # 获取图像的高度和宽度
        x = torch.randint(0, w - self.block_size[0], (1,))
        y = torch.randint(0, h - self.block_size[1], (1,))
        img = img.clone()  # 使用clone来创建图像的副本
        img[:, y:y + self.block_size[1], x:x + self.block_size[0]] = 0  # 遮挡部分像素块
        return img


if __name__ == '__main__':
    from torchvision.transforms import Normalize
    # 使用自定义转换
    transform = transforms.Compose([
        PartialBlockMask(block_size=(4, 4)),  # 调整遮挡的像素块大小
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用均值和标准差来进行标准化
    ])
    from matplotlib import pyplot as plt

    for _ in range(6):
        # 加载图像并应用转换
        img = torch.randn(3, 10, 8)  # 这里假设输入图像是一个3通道的张量，尺寸为(3, 64, 64)
        transformed_img = transform(img)
        transformed_img = torch.clamp(transformed_img, 0, 1)
        img = transformed_img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.show()
