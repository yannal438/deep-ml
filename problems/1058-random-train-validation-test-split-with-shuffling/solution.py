import numpy as np
import random
def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    # Shuffle the rows of the dataset using to generate a random ordering of row indices
    n = data.shape[0]
    generate = np.random.default_rng(seed).permutation(n)
    shuffle_data = data[generate]
	# Compute split indices usig floor division via 'int()'
    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)
    
    # Slicing 
    train = shuffle_data[:train_end]
    validation = shuffle_data[train_end:validation_end]
    test = shuffle_data[validation_end:]
    return [train, validation, test]
