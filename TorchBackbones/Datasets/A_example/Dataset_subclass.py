import torch
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """
    TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    模型输入：Tensor数据，
    输出： Tensor数据。
    """

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print(tensor_dataset[1])
# 输出：(tensor([-1.0351, -0.1004,  0.9168]), tensor(0.4977))

# 获取数据集大小
print(len(tensor_dataset))
# 输出：4
