import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    predicted_probs_np = np.asarray(predicted_probs)
    true_labels_np = np.asarray(true_labels)
    predicted_clip = np.clip(predicted_probs_np, epsilon, 1.0 - epsilon)
    loss = -np.sum(true_labels_np * np.log(predicted_clip), axis=-1)
    return float(np.mean(loss))