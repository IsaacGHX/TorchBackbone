import random
import matplotlib.pyplot as plt


def show_square_pics(train_dataset, k=3, color_mode='color'):
    # 随机选择9张图片索引
    indices = random.sample(range(len(train_dataset)), k ** 2)

    # 根据索引取出图片和标签
    images = [train_dataset[i][0] for i in indices]
    labels = [train_dataset[i][1] for i in indices]

    # 展示图片
    fig, axs = plt.subplots(k, k, figsize=(k * 2 + 2, k * 2 + 2))
    for i, ax in enumerate(axs.flatten()):
        if color_mode == 'color':
            ax.imshow(images[i].permute(1, 2, 0))
        else:
            ax.imshow(images[i].squeeze(), cmap=color_mode)
        ax.set_title("Label: {}".format(labels[i]))
        ax.axis('off')
    plt.tight_layout()
    plt.show()
