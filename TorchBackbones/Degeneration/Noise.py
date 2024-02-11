import numpy as np


def RandomNoise(images,noise_factor = 0.1, scale = 0.5):
    images_noisy = images + noise_factor * np.random.normal(loc=0.0, scale=scale, size=images.shape)
    images_noisy = np.clip(images_noisy, 0.0, 1.0)

    return images_noisy

__all__ = ['RandomNoise']  # 列出你希望导入的对象
