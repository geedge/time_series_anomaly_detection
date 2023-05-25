from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate(y_true: list, y_pred: list) -> float:
    """
    F1评估方法
    :param y_true: 真实标签
    :param y_pred: 检测标签
    :return: f1、recall、precision
    """
    f1 = f1_score(y_true, y_pred)
    return f1