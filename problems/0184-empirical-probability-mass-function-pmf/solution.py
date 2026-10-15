from collections import Counter
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    if not empirical_pmf:
        return []
    total_element = len(samples)
    occurence_n = Counter(samples)
    pmf = []
    for valeur, occurence in occurence_n.items():
        probabilite = occurence / total_element
        pmf.append((valeur, probabilite))
        pmf.sort(key=lambda x: x[0])
    
    return pmf