import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    # Your code here
    flat_scores = []
    for item in y_scores:
        if isinstance(item, list):
            flat_scores.append(item[0])
        else:
            flat_scores.append(item)

    flat_true = []
    for item in y_true:
        if isinstance(item, list):
            flat_true.append(item[0])
        else:
            flat_true.append(item)
    y_bin = [1 if score >= threshold else 0 for score in flat_scores]

    vp = 0
    fp = 0
    vn = 0
    fn = 0
    for reel, predit in zip(y_true, y_bin):
        if reel == 1 and predit == 1:
            vp += 1
        elif reel == 0 and predit == 1:
            fp += 1
        elif reel == 0 and predit == 0:
            vn += 1
        elif reel == 1 and predit == 0:
            fn += 1
    
    if (vp + fp) == 0 or (vp + fn) == 0:
        return [0.0, 0.0]
    else:
        precision = vp /(vp + fp)
        recall =  vp / (vp + fn)
    return [round(precision, 4), round(recall, 4)]
    
    

    
    pass
