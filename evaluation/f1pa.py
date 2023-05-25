import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    该函数是point-adjust官方源码
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score =: The anomaly score
        label : The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred : if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def evaluate(y_true: list, y_pred: list) -> float:
    """
    F1PA评估方法，经过point adjust调整标签后再用F1评分
    :param y_true: 真实标签
    :param y_pred: 检测标签
    :return: 经过pa调整后的f1、recall、precision
    """
    y_true, y_pred = y_true.copy(), y_pred.copy()
    adjust_y_pred = adjust_predicts(score=np.array([0] * len(y_true)), label=y_true, pred=y_pred)
    f1 = f1_score(y_true, adjust_y_pred)
    return f1

