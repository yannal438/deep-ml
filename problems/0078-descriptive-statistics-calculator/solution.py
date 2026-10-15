import numpy as np
import statistics 

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    moyenne = np.mean(data)
    mediane = np.median(data)
    mode = statistics.mode(data)
    variance = np.var(data)
    ecart_type = np.std(data)
    Q1 = np.percentile(data, 25)
    Q2 = np.percentile(data, 50)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1 
    return {
        'mean': moyenne,
        'median':mediane,
        'mode':mode,
        'variance':variance,
        'standard_deviation':ecart_type,
        '25th_percentile':Q1,
        '50th_percentile':Q2,
         '75th_percentile':Q3,
        'interquartile_range':IQR
    }
    