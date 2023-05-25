import math
import numpy as np


def evaluate(y_true: [int], y_pred: [int], pos_label: int = 1, max_segment: int = 0) -> float:
    """
    基于异常段计算F值

    :param y_true: 真实标签
    :param y_pred: 检测标签
    :param pos_label: 检测的目标数值，即指定哪个数为异常数值
    :param max_segment: 异常段最大长度
    :return: 段F值
    """
    p_tad = precision_tad(y_true=y_true, y_pred=y_pred, pos_label=pos_label, max_segment=max_segment)
    r_tad = recall_tad(y_true=y_true, y_pred=y_pred, pos_label=pos_label, max_segment=max_segment)
    score = 0
    if p_tad and r_tad:
        score = 2 * p_tad * r_tad / (p_tad + r_tad)
    return score


def recall_tad(y_true: [int], y_pred: [int], pos_label: int = 1, max_segment: int = 0) -> float:
    """
    基于异常段计算召回率

    :param y_true: 真实标签
    :param y_pred: 检测标签
    :param pos_label: 检测的目标数值，即指定哪个数为异常数值
    :param max_segment: 异常段最大长度
    :return: 段召回率
    """
    if max_segment == 0:
        max_segment = get_max_segment(y_true=y_true, pos_label=pos_label)
    score = tp_count(y_true, y_pred, pos_label=pos_label, max_segment=max_segment)
    return score


def precision_tad(y_true: [int], y_pred: [int], pos_label: int = 1, max_segment: int = 0) -> float:
    """
    基于异常段计算精确率

    :param y_true: 真实标签
    :param y_pred: 检测标签
    :param pos_label: 检测的目标数值，即指定哪个数为异常数值
    :param max_segment: 异常段最大长度
    :return: 段精确率
    """
    if max_segment == 0:
        max_segment = get_max_segment(y_true=y_true, pos_label=pos_label)
    score = tp_count(y_pred, y_true, pos_label=pos_label, max_segment=max_segment)
    return score


def tp_count(y_true: [int], y_pred: [int], max_segment: int = 0, pos_label: int = 1) -> float:
    """
    计算段的评分，交换y_true和y_pred可以分别表示召回率与精确率。

    :param y_true: 真实标签
    :param y_pred: 检测标签
    :param pos_label: 检测的目标数值，即指定哪个数为异常数值
    :param max_segment: 异常段最大长度
    :return: 分数
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred should have the same length.")
    neg_label = 1 - pos_label
    position = 0
    tp_list = []
    if max_segment == 0:
        raise ValueError("max segment length is 0")
    while position < len(y_true):
        if y_true[position] == neg_label:
            position += 1
            continue
        elif y_true[position] == pos_label:
            start = position
            while position < len(y_true) and y_true[position] == pos_label and position - start < max_segment:
                position += 1
            end = position
            true_window = [weight_line(i/(end-start)) for i in range(end-start)]
            true_window = softmax(true_window)
            pred_window = np.array(y_pred[start:end])
            pred_window = np.where(pred_window == pos_label, true_window, 0)
            tp_list.append(sum(pred_window))
        else:
            raise ValueError("label value must be 0 or 1")
    score = sum(tp_list) / len(tp_list) if len(tp_list) > 0 else 0
    return score


def weight_line(position: float) -> float:
    """
    按照权重曲线，给不同位置的点赋值

    :param position: 点在段中的相对位置，取值范围[0,1]
    :return: 权重值
    """
    if position < 0 or position > 1:
        raise ValueError(f"point position in segment need between 0 and 1, {position} is error position")
    sigma = 1 / (1 + math.exp(10*(position-0.5)))
    return sigma


def softmax(x: [float]) -> [float]:
    """
    softmax函数
    :param x: 一个异常段的数据
    :return: 经过softmax的一段数据
    """
    ret = np.exp(x)/np.sum(np.exp(x), axis=0)
    return ret.tolist()


def get_max_segment(y_true: [int], pos_label: int = 1) -> int:
    """
    获取最大的异常段的长度
    :param y_true: 真实标签
    :param pos_label: 异常标签的取值
    :return: 最大长度
    """
    max_num, i = 0, 0
    neg_label = 1 - pos_label
    while i < len(y_true):
        if y_true[i] == neg_label:
            i += 1
            continue
        elif y_true[i] == pos_label:
            start = i
            while i < len(y_true) and y_true[i] == pos_label:
                i += 1
            end = i
            max_num = max(max_num, end-start)
        else:
            raise ValueError("label value must be 0 or 1")
    return max_num


if __name__ == "__main__":
    
    # y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #           0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #           0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    import pandas as pd
    data = pd.read_csv("../records/2023-04-10_10-30-27/detection_result/MtadGatAtt_SWAT.csv")
    y_true = data["true"].tolist()
    y_pred = data["ftad"].tolist()

    print(evaluate(y_true, y_pred))
    # print(precision_tad(y_true, y_pred))
    # print(recall_tad(y_true, y_pred))
    # from sklearn.metrics import f1_score, precision_score, recall_score
    # print(f1_score(y_true, y_pred))
    # print(precision_score(y_true, y_pred))
    # print(recall_score(y_true, y_pred))
