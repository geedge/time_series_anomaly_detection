import torch.nn as nn
import torch.utils.data as tud
import torch


class Model(nn.Module):
    def __init__(self, customs: dict, dataloader: tud.DataLoader = None):
        """
        :param customs: 自定义参数，内容取自于config.ini文件的[CustomParameters]部分。
        :param dataloader: 数据集初始化完成的dataloader。在自定义的预处理方法文件中，可以增加内部变量或者方法，提供给模型。
                           例如：模型初始化需要数据的维度数量，可通过n_features = dataloader.dataset.train_inputs.shape[-1]获取
                                或在预处理方法的MyDataset类中，定义self.n_features = self.train_inputs.shape[-1]，
                                通过n_features = dataloader.dataset.n_features获取
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        :param x: 模型的输入，在本工具中为MyDataset类中__getitem__方法返回的三个变量中的第一个变量。
        :return: 模型的输出，可以自定义
        """
        return None

    def loss(self, x, y_true, epoch: int = None, device: str = "cpu"):
        """
        计算loss。注意，计算loss时如采用torch之外的库计算会造成梯度截断，请全部使用torch的方法
        :param x: 输入数据
        :param y_true: 真实输出数据
        :param epoch: 当前是第几个epoch
        :param device: 设备，cpu或者cuda
        :return: loss值
        """
        y_pred = self.forward(x)  # 模型的输出
        loss = torch.Tensor([1])  # 示例，请修改
        loss.backward()
        return loss.item()

    def detection(self, x, y_true, epoch: int = None, device: str = "cpu"):
        """
        检测方法，可以输出异常的分数，也可以输出具体的标签。
        如输出异常分数，则后续会根据异常分数自动划分阈值，高于阈值的为异常，自动赋予标签；如输出标签，则直接进行评估。
        :param x: 输入数据
        :param y_true: 真实输出数据
        :param epoch: 当前是第几个epoch
        :param device: 设备，cpu或者cuda
        :return: score，label。如选择输出异常的分数，则输出score，label为None；如选择输出标签，则输出label，score为None。
                 score的格式为torch的Tensor格式，尺寸为[batch_size]；label的格式为torch的IntTensor格式，尺寸为[batch_size]
        """
        y_pred = self.forward(x)  # 模型的输出
        return None, None