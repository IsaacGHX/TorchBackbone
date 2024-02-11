import numpy as np
import matplotlib.pyplot as plt
import cv2

# 创建一个多边形节点的list
points = [[[0, 0], [56, 20], [88, 100], [0, 100]], [[10, 10], [156, 120], [188, 200], [0, 200]]]
points = np.array(points)

# 创建一个与原图像大小相同的掩膜图像
mask = np.zeros((300, 300), dtype=np.uint8)

# 使用fillPoly()函数来填充掩膜图像
cv2.fillPoly(mask, points, (255, 0, 0))

# 保存掩膜图像
cv2.imwrite('mask.png', mask)

import cv2

# 创建一个多边形节点的list
points1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
points2 = np.array([[200, 0], [300, 0], [300, 100], [200, 100]])
pp = np.vstack([points1, points2])
color = tuple(np.random.randint(0, 255, size=3))  # Random color for example
color = tuple([int(c * (1 - 0.5)) for c in color])  # Apply transparency
# 创建一个与原图像大小相同的掩膜图像
mask = np.zeros((400, 400), dtype=np.uint8)

# 使用fillPoly()函数来填充掩膜图像
cv2.fillPoly(mask, [points1, points2], color)

# 将掩膜图像转换为2通道图像
mask = mask[:, :]

# 展示掩膜图像
plt.imshow(mask)
plt.show()
