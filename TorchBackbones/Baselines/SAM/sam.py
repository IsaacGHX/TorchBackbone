from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# image = cv2.imread('D:/datasets/WIDER_val/WIDER_val/images/0--Parade/0_Parade_marchingband_1_74.jpg')
image = cv2.imread('../HQ_SAM/val_1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

MODE = "default"
# MODE = "vit_l"
# MODE = "vit_b"
# MODE = "vit_tiny"
#
model_path = "D:/datasets/Models/SAM/sam_vit_h_4b8939.pth"
# model_path = "D:/Downloads/sam_vit_l_0b3195.pth"
# model_path = "D:/Downloads/sam_vit_b_01ec64.pth"
# model_path = "D:/Downloads/sam_hq_vit_tiny.pth"

device = "cuda"
sam = sam_model_registry[MODE](checkpoint=model_path)
# sam_vit_h_4b8939.pth 是预训练的默认权重，需要单独下载
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


import time

t0 = time.time()
masks = mask_generator.generate(image)
print(len(masks))
print(masks[5]['segmentation'].sum())
print(masks[6]['segmentation'].sum())
print(time.time() - t0)
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.title(MODE)
plt.show()
