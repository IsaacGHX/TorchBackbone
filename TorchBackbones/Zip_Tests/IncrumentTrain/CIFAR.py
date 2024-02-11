from ZentaiCheck.devices_confirm import *

from Models.BackBones.ResNets.ResNet18_refine import refine_resnet18

model = refine_resnet18(n_classes=10, in_channels=3)

from torchinfo import summary

# summary(model, input_size=(64, 3, 32, 32))

model.to(device)

from Datasets.Classification.CIFAR10 import *




