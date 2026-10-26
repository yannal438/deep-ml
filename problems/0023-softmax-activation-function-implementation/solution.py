import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores_stable = scores - np.max(scores)
    soft_n = np.exp(scores_stable)
    soft_d = np.sum(soft_n)
    soft = soft_n / soft_d
    return soft.tolis