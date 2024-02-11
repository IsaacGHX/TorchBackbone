from torch import nn
import torch
from tqdm import tqdm
from torch.nn.functional import one_hot
from torch import optim


def train(model, train_loader, val_loader, batch_size, criterion=nn.CrossEntropyLoss(label_smoothing=0.02),
          num_epochs=3):
    best_val_loss = float("inf")
    from lion_pytorch import Lion
    # optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-8)
    optimizer = Lion(model.parameters(), lr=5e-5, weight_decay=1e-8)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        # 训练
        model.train()
        train_loss = 0.0
        # Initialize progress bar
        progress_bar = tqdm(train_loader, leave=True)
        i = 1

        for images, target in progress_bar:
            images = images.cuda()
            target = target.cuda()
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)  # 在代码中加入这行实现梯度裁剪
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': train_loss / batch_size / i})
            i += 1

        train_loss = train_loss / len(train_loader)
        train_loss_history.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        accurate = 0

        progress_bar = tqdm(val_loader, leave=False)
        with torch.no_grad():
            for images, target in progress_bar:
                images = images.cuda()
                target = target.cuda()
                output = model(images)
                loss = criterion(output, target)
                val_loss += loss.item()
                accurate += (output.argmax(1) == target).sum()
        val_loss = val_loss / len(val_loader)
        val_loss_history.append(val_loss)

        # 打印训练进度
        print("Epoch{:d} ----------------------- train_loss: {:.4f}, val_loss: {:.4f}, accuracy: {:.4f}%".format(
            epoch + 1, train_loss, val_loss, accurate / len(val_loader) / batch_size * 100))

        # 保存最优模型
        if abs(val_loss) < abs(best_val_loss):
            best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

        history = {"train_loss": train_loss_history, "val_loss": val_loss_history}
    return history
