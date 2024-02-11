import os
from tqdm import tqdm


def get_data(img_dir, anno_file):
    with open(anno_file) as f:
        lines = f.readlines()

    img_paths = []
    img_boxes = []
    img_labels = []

    for line in tqdm(lines):
        line = line.strip()
        if line.endswith('.jpg'):
            img_name = line
            img_path = os.path.join(img_dir, img_name)
            img_paths.append(img_path)
            img_boxes.append([])
            img_labels.append([])
        elif line.isdigit():
            num_boxes = int(line)
        else:
            box = line.split()
            img_boxes[-1].append([int(x) for x in box[:4]])
            img_labels[-1].append([int(x) for x in box[4:]])

    return img_paths, img_boxes, img_labels


__all__ = ['get_data']

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    data_path = "D:/datasets/WIDER_train/images"
    label_path = "D:/datasets/wider_face_split/wider_face_train_bbx_gt.txt"
    img_paths, img_boxes, img_labels = get_data(data_path, label_path)
    print(len(img_paths))

    # 假设img_paths, img_boxes已通过get_data()获取

    img_idx = 33  # 选择要显示的图像索引
    img_path = img_paths[img_idx]
    bboxes = img_boxes[img_idx]
    print(img_path)
    print(len(bboxes))
    print(bboxes)

    # 读取图像并显示
    img = plt.imread(img_path)
    print(img.shape)
    plt.imshow(img)

    # 为每个bbox添加rectangle patch
    ax = plt.gca()
    for bbox in bboxes:
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

    cropped_imgs = []
    for bbox in bboxes:
        x, y, w, h = bbox
        crop_img = img[y:y + h, x:x + w]
        cropped_imgs.append(crop_img)

    # 展示每个裁剪区域
    fig, axs = plt.subplots(ncols=len(cropped_imgs))
    plt.axis('off')
    for i, crop_img in enumerate(cropped_imgs):
        axs[i].imshow(crop_img)
        axs[i].axis("off")
    plt.show()
