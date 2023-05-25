from .affiliation_bin.generics import convert_vector_to_events
from .affiliation_bin.metrics import pr_from_events


def evaluate(y_true: list, y_pred: list) -> float:
    """
    F1PA评估方法，经过point adjust调整标签后再用F1评分
    :param y_true: 真实标签
    :param y_pred: 检测标签
    :return: affiliation的三个score
    """
    true, pred = y_true.copy(), y_pred.copy()
    events_pred = convert_vector_to_events(pred)
    events_gt = convert_vector_to_events(true)
    Trange = (0, len(pred))

    res = pr_from_events(events_pred, events_gt, Trange)
    aff_precision = res["precision"]
    aff_recall = res["recall"]
    if aff_recall == 0 or aff_precision == 0:
        return 0
    aff_f1 = 2 * aff_precision * aff_recall / (aff_precision + aff_recall)
    return aff_f1