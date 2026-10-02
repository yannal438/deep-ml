def confusion_matrix(data):
    vp = 0
    fp = 0
    vn = 0
    fn = 0

    # CORRECTION : On extrait dans l'ordre de l'énoncé [y_true, y_pred]
    for y_true, y_pred in data:
        if y_true == 1 and y_pred == 1:
            vp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            vn += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
            
    # Renvoie la liste de listes 2x2 demandée
    return [[vp, fn],[fp, vn]]

