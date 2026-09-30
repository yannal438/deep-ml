import numpy as np

def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # --- STANDARDISATION ---
    moyenne = np.mean(data, axis=0)
    ecart_type = np.std(data, axis=0)
    numerateur = data - moyenne
    standardized_data = numerateur / ecart_type

    # --- NORMALISATION ---
    val_min = np.min(data, axis=0)
    val_max = np.max(data, axis=0)
    num = (data - val_min)
    deno = (val_max - val_min)  # CORRECTION : max - min
    normalized_data = num / deno
    
    # --- RETOUR & ARRONDI ---
    return np.round(standardized_data, 4), np.round(normalized_data, 4)
