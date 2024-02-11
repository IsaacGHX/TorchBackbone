import pytorch_lightning as pl
import torch.nn.functional as F
import torch

from Models.BackBones.ResNets.ResNet18_refine import refine_resnet18

model = refine_resnet18(10,in_channels=1)


# 定义PyTorch Lightning 模型
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.all = model

    def forward(self, x):
        out = self.all(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# 实例化模型
lt_model = LitModel()

from Datasets.Classification.MNIST import *

# 训练
trainer = pl.Trainer(gpus=1)
trainer.fit(lt_model, train_loader, test_loader)
