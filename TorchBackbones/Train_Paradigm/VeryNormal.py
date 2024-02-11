import torch.nn as nn
import torch
from tqdm import tqdm


def train(model, train_dataloader, val_dataloader, loss, optimizer, batch_size, lr, input_size,device,
          epochs=1, weight_decay=0., summarize=True, save =False):
    # 创建网络模型
    model = model.to(device)

    if summarize:
        from torchinfo import summary
        data, target = next(iter(train_dataloader))
        summary(model, input_size=input_size)

    # 损失函数
    loss = loss
    if weight_decay != 0.:
        optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optimizer(model.parameters(), lr=lr)

    i = 1  # 用于绘制测试集的tensorboard

    # 开始循环训练
    for epoch in range(epochs):
        num_time = 0  # 记录看看每轮有多少次训练
        print('开始第{}轮训练'.format(epoch + 1))
        model.train()  # 也可以不写，规范的话是写，用来表明训练步骤
        for data in tqdm(train_dataloader):
            # 数据分开 一个是图片数据，一个是真实值
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            # 拿到预测值
            output = model(imgs)
            # 计算损失值
            loss_in = loss(output, targets)
            # 优化开始~ ~ 先梯度清零
            optimizer.zero_grad()
            # 反向传播+更新
            loss_in.backward()
            optimizer.step()
            num_time += 1

        sum_loss = 0  # 记录总体损失值

        # 每轮训练完成跑一下测试数据看看情况
        accurate = 0
        model.eval()  # 也可以不写，规范的话就写，用来表明是测试步骤
        with torch.no_grad():
            for data in val_dataloader:
                # 这里的每一次循环 都是一个minibatch  一次for循环里面有dim*2个数据。
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = model(imgs)
                loss_in = loss(output, targets)

                sum_loss += loss_in
                # print('这里是output',output)
                accurate += (output.argmax(1) == targets).sum()

        print('第{}轮验证集的正确率:{:.2f}%'.format(epoch + 1, accurate / len(val_dataloader) / batch_size * 100))

        i += 1

        if save:
            torch.save(model, '../_CheckPoints/Train_Paradigm/model_{}.pth'.format(epoch + 1))
            print("第{}轮模型训练数据已保存".format(epoch + 1))


if __name__ == '__main__':
    print("CUDA is: ",torch.cuda.is_available())
    from Datasets.Classification.MNIST import *

    loss = nn.CrossEntropyLoss(label_smoothing=0.05).cuda()
    from lion_pytorch import Lion

    from Models.BackBones.ResNets.ResNet18_refine import refine_resnet18

    model = refine_resnet18(n_classes=10, in_channels=1)

    train(model, train_loader, val_loader, loss, Lion, batch_size, lr=1e-3,
          epochs=3, weight_decay=1e-6, summarize=True, input_size=(1, 1, 28, 28))
