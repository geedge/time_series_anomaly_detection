# -*- coding:utf-8 -*-
import configparser
from daemon import Daemon
import sys
from torch.cuda import is_available
from loguru import logger
from datetime import datetime
from utils import create_all_dataloader, train_and_test_model
import method
import gc
import traceback
import pandas
import os


class Config(configparser.ConfigParser):
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, optionstr):
        return optionstr

    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


class Main(Daemon):
    def __init__(self, pidfile):
        super(Main, self).__init__(pidfile=pidfile)
        current = datetime.now()
        self.start_time = current.strftime("%Y-%m-%d_%H-%M-%S")

        if len(sys.argv) == 1 or sys.argv[1] == "start":
            self.run()
        elif sys.argv[1] == "stop":
            self.stop()
        elif sys.argv[1] == "daemon":
            self.start()
        else:
            print("Input format error. Please input: python3 main.py start|stop|daemon")
            sys.exit(0)

    def run(self):
        # 读取配置文件参数
        cf = Config()
        cf.read("./config.ini", encoding='utf8')
        cf_dict = cf.as_dict()
        # 读取数据集名称
        dataset_names = cf.get("Dataset", "name")
        datasets = dataset_names.split(",")
        datasets = [name.strip() for name in datasets]
        # 读取模型名称
        model_names = cf.get("Method", "name")
        models = model_names.split(",")
        models = [name.strip() for name in models]
        # 读取预处理方法
        preprocess_name = cf.get("Preprocess", "name")
        # 读取评估方法
        evaluation_names = cf.get("Evaluation", "name")
        evaluations = evaluation_names.split(",")
        evaluations = [name.strip() for name in evaluations]
        # 读取模型参数文件路径
        model_path = cf_dict["ModelPath"]
        # 读取训练参数
        train = cf.getboolean("BaseParameters", "train")
        epochs = cf.getint("BaseParameters", "epochs")
        batch_size = cf.getint("BaseParameters", "batch_size")
        learning_rate = cf.getfloat("BaseParameters", "learning_rate")
        device = cf.get("BaseParameters", "device")
        if device == "auto":
            device = 'cuda:0' if is_available() else 'cpu'
        # 读取自定义参数
        customs = cf_dict["CustomParameters"]

        # 建立本次实验记录的路径
        os.makedirs(f"./records", exist_ok=True)
        os.makedirs(f"./records/{self.start_time}", exist_ok=True)
        os.makedirs(f"./records/{self.start_time}/detection_result", exist_ok=True)
        os.makedirs(f"./records/{self.start_time}/model", exist_ok=True)

        # 初始化日志
        logger.add(f"./records/{self.start_time}/log",
                   level='DEBUG',
                   format='{time:YYYY-MM-DD HH:mm:ss} - {level} - {file} - {line} - {message}',
                   rotation="100 MB")

        # 核心程序
        self.core(models, datasets, preprocess_name, evaluations, model_path, train, epochs, batch_size, learning_rate,
                  device, customs)
        logger.info(f"实验结束，关闭进程")

    def core(self, models: [str], datasets: [str], preprocess_name: str, evaluations: [str], model_path: {}, train: bool,
             epochs: int, batch_size: int, learning_rate: float, device: str, customs: {}):
        """
        初始化数据集与模型，并开始训练与测试
        :param models: 训练的模型名称，可包含多个
        :param datasets: 使用的数据集名称，可包含多个
        :param preprocess_name: 预处理方法名称
        :param evaluations: 评估方法名称，可包含多个
        :param model_path: 需要加载模型参数的路径，可包含多个
        :param train: 是否训练，如果为False，则仅测试模型
        :param epochs: 总训练轮数
        :param batch_size: batch的尺寸
        :param learning_rate: 学习率
        :param device: 设备
        :param customs: 自定义参数
        """
        logger.info(f"加载数据集")
        try:
            # 初始化所有数据集
            all_dataloader = create_all_dataloader(datasets=datasets, input_size=int(customs["input_size"]),
                                                   output_size=int(customs["output_size"]), step=int(customs["step"]),
                                                   batch_size=batch_size, time_index=customs["time_index"] == "true",
                                                   del_column_name=customs["del_column_name"] == "true",
                                                   preprocess_name=preprocess_name)
        except RuntimeError:
            logger.error(traceback.format_exc())
            return

        # 开始训练与测试

        for model_name in models:
            try:
                logger.info(f"------------华丽丽的分界线：{model_name} 实验开始------------")
                for i in range(len(all_dataloader)):
                    dataloader = all_dataloader[i]
                    all_score = {}
                    for sub_dataloader in dataloader:
                        dataset_name = sub_dataloader["dataset_name"]
                        normal_dataloader = sub_dataloader["normal"]
                        attack_dataloader = sub_dataloader["attack"]
                        logger.info(f"初始化模型 {model_name}")
                        model = eval(f"method.{model_name}.Model")(customs=customs, dataloader=normal_dataloader)
                        model = model.to(device)
                        logger.info(f"模型初始化完成")
                        pth_name = model_path[f"{model_name}_{dataset_name}"] if f"{model_name}_{dataset_name}" \
                                                                                 in model_path else None
                        best_score, best_detection = train_and_test_model(start_time=self.start_time, epochs=epochs,
                                                                          normal_dataloader=normal_dataloader,
                                                                          attack_dataloader=attack_dataloader,
                                                                          model=model, evaluations=evaluations,
                                                                          device=device, lr=learning_rate,
                                                                          model_path=pth_name, train=train)
                        # 保存最佳检测结果的标签为csv文件
                        best_detection = pandas.DataFrame(best_detection)
                        best_detection.to_csv(f"./records/{self.start_time}/detection_result/{model_name}_{dataset_name}.csv", index=False)
                        for evaluation_name in evaluations:
                            if evaluation_name not in all_score:
                                all_score[evaluation_name] = []
                            all_score[evaluation_name].append(best_score[evaluation_name])
                        gc.collect()
                    logger.info(f"------------------------")
                    logger.info(f"{model_name} / {datasets[i]} 实验完毕")
                    for evaluation_name in all_score:
                        logger.info(f"{evaluation_name}: {'{:.3f}'.format(sum(all_score[evaluation_name]) / len(all_score[evaluation_name]))}")
                    logger.info(f"------------------------")
                logger.info(f"------------华丽丽的分界线：{model_name} 实验结束------------")
            except RuntimeError:
                logger.error(traceback.format_exc())
        return


if __name__ == '__main__':
    pidpath = "/tmp/command_detection.pid"
    app = Main(pidpath)

