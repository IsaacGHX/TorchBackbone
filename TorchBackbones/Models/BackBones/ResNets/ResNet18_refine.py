from Models.BackBones.ResNets.ResNet import ResNet18
import torch
import torch.nn as nn


def refine_resnet18(n_classes,in_channels):
    model = ResNet18()  # 得到预训练模型
    model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, n_classes)  # 将最后的全连接层修改
    return model

__all__=['refine_resnet18']