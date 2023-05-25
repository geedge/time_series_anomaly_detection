from torch.utils.data import Dataset
from torch import float32, Tensor
from numpy import array, where


class MyDataset(Dataset):
    def __init__(self, name: str, train_path: str = None, test_path: str = None, input_size: int = 1,
                 output_size: int = 1, step: int = 1, mode: str = 'train', time_index: bool = True,
                 del_column_name: bool = True):
        """
        可以将csv文件批量转成tensor
        :param name: 数据集名称。
        :param train_path: 训练数据集路径。
        :param test_path: 测试数据集路径。
        :param input_size: 输入数据长度。
        :param output_size: 输出数据长度。
        :param step: 截取数据的窗口移动间隔。
        :param mode: train或者test，用于指示使用训练集数据还是测试集数据。
        :param time_index: True为第一列是时间戳，False为不。
        :param del_column_name: 文件中第一行为列名时，使用True。
        """
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.del_column_name = del_column_name
        self.step = step
        self.mode = mode
        self.time_index = time_index
        self.train_inputs, self.train_labels, self.train_outputs, self.test_inputs, self.test_labels, self.test_outputs\
            = self.parse_data(train_path, test_path)
        self.train_inputs = Tensor(self.train_inputs).to(float32) if self.train_inputs is not None else None
        self.train_labels = Tensor(self.train_labels).to(float32) if self.train_labels is not None else None
        self.train_outputs = Tensor(self.train_outputs).to(float32) if self.train_outputs is not None else None
        self.test_inputs = Tensor(self.test_inputs).to(float32) if self.test_inputs is not None else None
        self.test_labels = Tensor(self.test_labels).to(float32) if self.test_labels is not None else None
        self.test_outputs = Tensor(self.test_outputs).to(float32) if self.test_outputs is not None else None

    def parse_data(self, train_path: str = None, test_path: str = None):
        if train_path is None and test_path is None:
            raise ValueError("train_path is None and test_path is None.")

        mean = None
        deviation = None
        train_data_input, train_label, train_data_output = None, None, None
        test_data_input, test_label, test_data_output = None, None, None

        # 读取训练集数据
        if train_path:
            train_data = []
            train_label = []
            with open(train_path, 'r', encoding='utf8') as f:
                if self.del_column_name is True:
                    data = f.readlines()[1:]
                else:
                    data = f.readlines()
            train_data.extend([list(map(float, line.strip().split(','))) for line in data])
            train_label.extend([0 for _ in data])
            train_np = array(train_data)
            if self.time_index:
                train_np[:, 0] = train_np[:, 0] % 86400
            mean = train_np.mean(axis=0)  # 计算平均数
            deviation = train_np.std(axis=0)  # 计算标准差
            deviation = where(deviation != 0, deviation, 1)
            train_np = (train_np - mean) / deviation  # 标准化
            train_data = train_np.tolist()
            train_data_input, train_data_output, train_label = self.cut_data(train_data, train_label)

        # 读取测试集数据
        if test_path:
            test_data = []
            test_label = []
            with open(test_path, 'r', encoding='utf8') as f:
                if self.del_column_name is True:
                    data = f.readlines()[1:]
                else:
                    data = f.readlines()
            test_data.extend([list(map(float, line.strip().split(',')))[:-1] for line in data])
            test_label.extend([int(line.strip().split(',')[-1]) for line in data])
            test_np = array(test_data)
            if self.time_index:
                test_np[:, 0] = test_np[:, 0] % 86400
            # mean = test_np.mean(axis=0)  # 计算平均数
            # deviation = test_np.std(axis=0)  # 计算标准差
            # deviation = where(deviation != 0, deviation, 1)
            test_np = (test_np - mean) / deviation  # 标准化
            test_data = test_np.tolist()
            # 自动判断是否需要反转标签。异常标签统一认为是1，当异常标签超过一半时，需反转标签
            if sum(test_label) > 0.5*len(test_label):
                test_label = (1-array(test_label)).tolist()
            test_data_input, test_data_output, test_label = self.cut_data(test_data, test_label)

        return train_data_input, train_label, train_data_output, test_data_input, test_label, test_data_output

    def cut_data(self, data: [[float]], label: [int]):
        n = 0
        input_data, output_data, anomaly_label = [], [], []
        while n + self.input_size + self.output_size <= len(data):
            input_data.append(data[n: n + self.input_size])
            output_data.append(data[n + self.input_size: n + self.input_size + self.output_size])
            anomaly_label.append([max(label[n + self.input_size: n + self.input_size + self.output_size])])
            n = n + self.step
        return input_data.copy(), output_data.copy(), anomaly_label.copy()

    def __len__(self):
        if self.mode == 'train':
            return self.train_inputs.shape[0]
        elif self.mode == 'test':
            return self.test_inputs.shape[0]

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.train_inputs[idx], self.train_labels[idx], self.train_outputs[idx]
        elif self.mode == 'test':
            return self.test_inputs[idx], self.test_labels[idx], self.test_outputs[idx]


if __name__ == "__main__":
    app = MyDataset('../dataset/SWAT/train.csv', test_path='../dataset/SWAT/test.csv', input_size=3)
    print(app)
