"""
模板文件，有自定义预处理方法可以通过编辑本文件实现数据集预处理。
编辑完成以后，请将文件名修改，写入config.ini，并在同级目录下init文件添加本文件
"""


from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, train_path: str = None, test_path: str = None, input_size: int = 1, output_size: int = 1,
                 step: int = 1, mode: str = 'train', del_time: bool = True, del_column_name: bool = True,
                 reverse_label: bool = True):
        """
        可以将csv文件批量转成tensor

        注意：必须包含以下变量或方法。
        变量：self.train_inputs、self.train_labels、self.train_outputs
             self.test_inputs、self.test_labels、self.test_outputs、self.mode
        方法：__len__()、__getitem__()

        :param train_path: str类型。训练数据集路径。
        :param test_path: str类型。测试数据集路径。
        :param input_size: int类型。输入数据长度。
        :param output_size: int类型。输出数据长度。
        :param step: int类型。截取数据的窗口移动间隔。
        :param mode: str类型。train或者test，用于指示使用训练集数据还是测试集数据。
        :param del_time: bool类型。True为删除时间戳列，False为不删除。
        :param del_column_name: bool类型。文件中第一行为列名时，使用True。
        :param reverse_label: bool类型。转化标签，0和1互换。标签统一采用正常为0异常为1的格式，若原文件中不符和该规定，使用True。
        """

        self.mode = mode
        self.train_inputs = None # 训练时的输入数据，Tensor格式，尺寸为[N,L,D]。N表示训练数据的数量，L表示每条数据的长度（由多少个时间点组成的数据），D表示数据维度数量
        self.train_labels = None # 训练时的数据标签，Tensor格式，尺寸为[N,1]。
        self.train_outputs = None # 训练时的输出数据，Tensor格式，尺寸为[N,L,D]。
        self.test_inputs = None # 测试时的输入数据，Tensor格式，尺寸为[N,L,D]。
        self.test_labels = None # 测试时的数据标签，Tensor格式，尺寸为[N,1]。
        self.test_outputs = None # 测试时的输出数据，Tensor格式，尺寸为[N,L,D]。

    def __len__(self):
        """
        提供数据集长度
        :return: 测试集或者训练集数据长度N
        """
        if self.mode == 'train':
            return self.train_inputs.shape[0]
        elif self.mode == 'test':
            return self.test_inputs.shape[0]

    def __getitem__(self, idx):
        """
        获取数据
        :param idx: 数据序号
        :return: 对应的输入数据、标签、输出数据
        """
        if self.mode == 'train':
            return self.train_inputs[idx], self.train_labels[idx], self.train_outputs[idx]
        elif self.mode == 'test':
            return self.test_inputs[idx], self.test_labels[idx], self.test_outputs[idx]

