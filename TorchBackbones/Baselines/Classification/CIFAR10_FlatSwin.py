from Datasets.Classification.CIFAR10 import *
import torch
from torch import nn
from torch.utils.data import DataLoader

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=4,
                          num_workers=8, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=4,
                        num_workers=8, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from Train_Paradigm import fit_history

if __name__ == '__main__':
    from Models.BackBones.SwinTransformer.flatten_swin import FLattenSwinTransformer

    model = FLattenSwinTransformer(img_size=32, patch_size=4, in_chans=3, num_classes=10,
                                   embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12],
                                   window_size=4, mlp_ratio=3., qkv_bias=True, qk_scale=None,
                                   drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                   norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                   use_checkpoint=False,
                                   focusing_factor=3, kernel_size=3, attn_type='LLLL', ).cuda()

    history = fit_history.train(model, batch_size=batch_size, train_loader=train_loader, val_loader=val_loader,
                                num_epochs=20)

    import matplotlib.pyplot as plt

    # Extract data from history dictionary
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    # Configure plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
