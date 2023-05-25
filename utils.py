import os
import numpy
import torch.nn as nn
import torch
import torch.utils.data as tud
import preprocess
import evaluation
from loguru import logger
import time
from tqdm import tqdm


def create_dataloader(dataset_name: str, input_size: int = 1, output_size: int = 1, step: int = 1, batch_size: int = 1,
                      time_index: bool = True, del_column_name: bool = True,
                      preprocess_name: str = "standardization") -> (tud.DataLoader, tud.DataLoader):
    """
    针对一个数据集，构建Dataloader
    :param dataset_name: 数据集名称
    :param input_size: 输入数据长度
    :param output_size: 输出数据长度
    :param step: 截取数据的窗口移动间隔
    :param batch_size: batch的大小
    :param time_index: True为第一列是时间戳，False为不
    :param del_column_name: 文件中第一行为列名时，使用True
    :param preprocess_name: 预处理方法
    :return: 训练数据与测试数据
    """
    ds = eval(f"preprocess.{preprocess_name}.MyDataset")(name=dataset_name.replace("/", "-"),
                                                         train_path=f"./dataset/{dataset_name}/train.csv",
                                                         test_path=f"./dataset/{dataset_name}/test.csv",
                                                         input_size=input_size, output_size=output_size,
                                                         step=step, time_index=time_index,
                                                         del_column_name=del_column_name)
    normal_dl = tud.DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    ds.mode = "test"
    attack_dl = tud.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)
    return normal_dl, attack_dl


def create_all_dataloader(datasets: [str], input_size: int = 1, output_size: int = 1, step: int = 1,
                          batch_size: int = 1, time_index: bool = True, del_column_name: bool = True,
                          preprocess_name: str = "standardization") -> [{}]:
    """
    对所有数据集构建dataloader
    :param datasets: 数据集列表
    :param input_size: 输入数据长度
    :param output_size: 输出数据长度
    :param step: 截取数据的窗口移动间隔
    :param batch_size: batch的大小
    :param time_index: True为第一列是时间戳，False为不。
    :param del_column_name: 文件中第一行为列名时，使用True
    :param preprocess_name: 预处理方法
    :return: 所有数据集的dataloader构建结果
    """
    all_dataloader = []
    for dataset_name in datasets:
        logger.info(f'开始建立 dataloader {dataset_name}')
        if "train.csv" in os.listdir(f"./dataset/{dataset_name}"):
            normal_dl, attack_dl = create_dataloader(dataset_name=dataset_name, input_size=input_size,
                                                     output_size=output_size, step=step, batch_size=batch_size,
                                                     time_index=time_index, del_column_name=del_column_name,
                                                     preprocess_name=preprocess_name)
            all_dataloader.append([{
                'dataset_name': dataset_name,
                'normal': normal_dl,
                'attack': attack_dl
            }])
        else:
            all_sub_dataloader = []
            for sub_dataset_dir in os.listdir(f"./dataset/{dataset_name}"):
                sub_dataset_name = f"{dataset_name}/{sub_dataset_dir}"
                normal_dl, attack_dl = create_dataloader(dataset_name=sub_dataset_name, input_size=input_size,
                                                         output_size=output_size, step=step, batch_size=batch_size,
                                                         del_time=del_time, del_column_name=del_column_name)
                all_sub_dataloader.append({
                    'dataset_name': sub_dataset_name.replace("/", "-"),
                    'normal': normal_dl,
                    'attack': attack_dl
                })
            all_dataloader.append(all_sub_dataloader)

        logger.info(f'完成建立 dataloader {dataset_name}')
    return all_dataloader


class EvaluationScore:
    def __init__(self, evaluations: [str], attack=1):
        """
        用于自动划分阈值并进行批量评估
        :param evaluations: 使用的评估方法名称（需在evaluation文件夹中进行定义）
        :param attack: 异常的标签，0 or 1
        """
        self.time = 0
        self.f1 = 0
        self.f1_pa = 0
        self.f_tad = 0
        self.attack = attack
        self.normal = 1 - attack
        self._total_y_loss = None
        self._total_label = None
        self._total_pred_label = None
        self.true_pred_df = None
        self.true_pred_dict = None
        self.evaluations = evaluations
        self.scores = {}

    def add_data(self, y_loss, true_label, pred_label=None):
        """
        添加每个batch的数据
        :param y_loss: 数据偏差
        :param true_label: 真实数据标签
        :param pred_label: 预测标签
        """
        if pred_label is not None:
            if self._total_label is None and self._total_pred_label is None:
                self._total_label = true_label
                self._total_pred_label = pred_label
            else:
                self._total_label = torch.cat([self._total_label, true_label], dim=0)
                self._total_pred_label = torch.cat([self._total_pred_label, pred_label], dim=0)
            return

        y_loss = y_loss.view(-1).cpu().detach().numpy()
        true_label = true_label.view(-1).cpu().detach().numpy()

        if self._total_y_loss is None and self._total_label is None:
            self._total_y_loss = y_loss
            self._total_label = true_label
            return
        self._total_y_loss = numpy.concatenate((self._total_y_loss, y_loss), axis=0)
        self._total_label = numpy.concatenate((self._total_label, true_label), axis=0)

    def best_threshold(self, true_label: list, y_loss: list) -> dict:
        ret = {}
        for func_name in self.evaluations:
            threshold_max = max(y_loss)
            threshold_min = 0
            best_threshold = 0
            for _ in range(5):
                threshold_list = [threshold_max - i * (threshold_max - threshold_min) / 10 for i in range(11)]
                f1_list = []
                for threshold_one in threshold_list:
                    prediction_loss = numpy.where(numpy.array(y_loss) > threshold_one, self.attack, self.normal)
                    f1 = eval(f"evaluation.{func_name}.evaluate")(y_true=true_label, y_pred=prediction_loss.tolist())
                    f1_list.append(f1)
                ind = f1_list.index(max(f1_list))
                best_threshold = threshold_list[ind]
                if ind == 0:
                    threshold_max = threshold_list[ind]
                    threshold_min = threshold_list[ind+1]
                elif ind == len(threshold_list)-1:
                    threshold_max = threshold_list[ind-1]
                    threshold_min = threshold_list[ind]
                else:
                    threshold_max = threshold_list[ind-1]
                    threshold_min = threshold_list[ind+1]
            ret[func_name] = best_threshold
        return ret

    def auto_threshold(self):
        if self._total_pred_label is not None:
            return
        self._total_y_loss[numpy.isnan(self._total_y_loss)] = 0
        self._total_y_loss = self._total_y_loss / max(self._total_y_loss)
        thresholds = self.best_threshold(
            self._total_label.reshape(-1).data.tolist(), self._total_y_loss.reshape(-1).data.tolist())
        self.true_pred_dict = {
            'true': self._total_label.squeeze().tolist()
        }
        for func_name in thresholds:
            self.true_pred_dict[func_name] = \
                numpy.where(self._total_y_loss > thresholds[func_name], self.attack, self.normal).squeeze().tolist()
        # self.true_pred_df = pandas.DataFrame(self.true_pred_dict)
        for func_name in self.true_pred_dict:
            if func_name == "true":
                continue
            self.scores[func_name] = self.get_score(func_name)

    def get_score(self, func_name):
        if self._total_pred_label is not None:
            return eval(f"evaluation.{func_name}.evaluate")(self._total_label.reshape(-1).tolist(),
                                                            self._total_pred_label.reshape(-1).tolist())
        return eval(f"evaluation.{func_name}.evaluate")(self._total_label.reshape(-1).tolist(),
                                                        self.true_pred_dict[f"{func_name}"])

    def __str__(self):
        res = ""
        for func_name in self.scores:
            res += f"{func_name}={self.scores[func_name]:.3f} "
        return res[:-1]


def train_model(epoch: int, optimizer: torch.optim, dataloader: tud.DataLoader, model: nn.Module,
                device: str = "cpu") -> (nn.Module, str):
    """
    训练模型
    :param epoch: 当前训练轮数
    :param optimizer: 优化器
    :param dataloader: 数据集
    :param model: 模型
    :param device: 训练设备使用cpu还是gpu
    :return: 训练完成的模型；训练完成需要输出的信息
    """
    model.train()
    avg_loss = []
    dataloader.dataset.mode = "train"
    start_time = time.time()
    with tqdm(total=len(dataloader), ncols=150) as _tqdm:
        _tqdm.set_description(f'（进度条部分不会写进本地日志）epoch:{epoch}，训练进度')
        for data in dataloader:
            x = data[0].to(device)
            y_true = data[2].to(device)
            optimizer.zero_grad()
            loss = model.loss(x=x, y_true=y_true, epoch=epoch, device=device)
            avg_loss.append(loss)
            optimizer.step()
            _tqdm.set_postfix(loss='{:.6f}'.format(sum(avg_loss) / len(avg_loss)))
            _tqdm.update(1)
    end_time = time.time()
    info = f"epoch={epoch}, average loss={'{:.6f}'.format(sum(avg_loss) / len(avg_loss))}, " \
           f"train time={'{:.1f}'.format(end_time-start_time)}s"
    return model, info


def test_model(dataloader: tud.DataLoader, model: nn.Module, evaluations: [str], device: str = "cpu") -> \
        (EvaluationScore, str):
    """
    测试模型
    :param dataloader: 数据集
    :param model: 模型
    :param device: 训练设备使用cpu还是gpu
    :return: 评估分数；测试完成需要输出的信息
    """
    es = EvaluationScore(evaluations)
    model.eval()
    dataloader.dataset.mode = "test"
    start_time = time.time()
    with tqdm(total=len(dataloader), ncols=150) as _tqdm:
        _tqdm.set_description(f'（进度条部分不会写进本地日志）测试进度')
        with torch.no_grad():
            for data in dataloader:
                x = data[0].to(device)
                y_true = data[2].to(device)
                label_true = data[1].int().to(device)
                y_loss, label_pred = model.detection(x=x, y_true=y_true, device=device)
                if label_pred is not None:
                    es.add_data(y_loss=None, true_label=label_true, pred_label=label_pred)
                else:
                    es.add_data(y_loss=y_loss, true_label=label_true, pred_label=None)
                _tqdm.update(1)
    end_time = time.time()
    es.auto_threshold()
    es_score = es.__str__().replace(" ", ", ")
    info = f"{es_score}, test time={'{:.1f}'.format(end_time-start_time)}s"
    return es, info


def train_and_test_model(start_time: str, epochs: int, normal_dataloader: tud.DataLoader, attack_dataloader: tud.DataLoader,
                         model: nn.Module, evaluations: [str], device: str = "cpu", lr: float = 1e-4,
                         model_path: str = None, train: bool = True) -> (dict, dict):
    """
    训练与测试
    :param start_time: 实验的开始时间。此处用于寻找存放路径。
    :param epochs: 总共训练轮数
    :param normal_dataloader: 训练数据集
    :param attack_dataloader: 测试数据集
    :param model: 模型
    :param evaluations: 评估方法
    :param device: 设备
    :param lr: 学习率
    :param model_path: 模型参数文件路径
    :param train: 是否训练，如果为否，则仅进行测试
    :return: 各个评估方法的最佳分数、各个评估方法最佳情况下的检测标签
    """
    dataset_name = normal_dataloader.dataset.name
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if model_path:
        try:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"模型参数文件{model_path}加载成功")
        except:
            logger.warning(f"模型参数文件{model_path}加载失败，将训练新模型")
    logger.info(f"模型：{model.name}，数据集：{dataset_name}，设备：{device}，训练开始")
    best_score = {}
    best_detection = {}
    if train:
        logger.info(f"模式：训练并测试")
        for epoch in range(1, epochs+1):
            model, train_info = train_model(epoch=epoch, optimizer=optimizer, dataloader=normal_dataloader, model=model,
                                            device=device)
            es, test_info = test_model(dataloader=attack_dataloader, model=model, evaluations=evaluations,
                                       device=device)
            logger.info(f"{train_info}, {test_info}")
            es_score = es.__str__().replace(" ", "_")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'./records/{start_time}/model/model={model.name}_dataset={dataset_name}_epoch={epoch}_{es_score}.pth')
            for func_name in es.scores:
                if func_name not in best_score or es.scores[func_name] > best_score[func_name]:
                    best_score[func_name] = es.scores[func_name]
                    best_detection[func_name] = es.true_pred_dict[func_name]
                    best_detection["true"] = es.true_pred_dict["true"]
    else:
        logger.info(f"模式：仅进行测试")
        es, test_info = test_model(dataloader=attack_dataloader, model=model, evaluations=evaluations, device=device)
        logger.info(test_info)
        for func_name in es.scores:
            best_score[func_name] = es.scores[func_name]
            best_detection[func_name] = es.true_pred_dict[func_name]
            best_detection["true"] = es.true_pred_dict["true"]
    return best_score, best_detection




