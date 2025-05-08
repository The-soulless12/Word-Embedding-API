import math
from typing import List, Tuple, Any
from .vec_func import vec_dot, vec_ew_add, vec_ew_mul, vec_ew_op, vec_ew_sub, vec_random, vec_smul, vec_softmax, vec_val

def mat_ew_add(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    # Addition élément par élément de deux matrices : X + Y
    return [vec_ew_add(row_x, row_y) for row_x, row_y in zip(X, Y)]

def mat_ew_sub(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    # Soustraction élément par élément de deux matrices : X - Y
    return [vec_ew_sub(row_x, row_y) for row_x, row_y in zip(X, Y)]

def mat_ew_mul(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    # Multiplication élément par élément de deux matrices : X * Y
    return [vec_ew_mul(row_x, row_y) for row_x, row_y in zip(X, Y)]

def mat_ew_op(X: List[List[float]], Y: List[List[float]], op) -> List[List[float]]:
    # Opération binaire élément par élément sur deux matrices X et Y
    return [vec_ew_op(row_x, row_y, op) for row_x, row_y in zip(X, Y)]

def mat_dot(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    # Produit matriciel de deux matrices : X @ Y
    Y_T = list(zip(*Y))
    return [[vec_dot(row_x, col_y) for col_y in Y_T] for row_x in X]

def mat_add_vec(X: List[List[float]], v: List[float], col: bool = False) -> List[List[float]]:
    # Ajoute un vecteur v à chaque ligne (ou colonne) de la matrice X
    if col:
        return [[X[i][j] + v[j] for j in range(len(v))] for i in range(len(X))]
    
    return [[X[i][j] + v[i] for j in range(len(X[0]))] for i in range(len(v))]

def mat_flatten(X: List[List[float]], col: bool = False) -> List[float]:
    # Aplatissement d’une matrice ligne par ligne (ou colonne par colonne si col=True
    X2 = zip(*X) if col else X
    return [x for X1 in X2 for x in X1]

def mat_random(n: int, m:int) -> List[List[float]]:
    # Génère une matrice aléatoire de taille (n × m)
    return [vec_random(m) for _ in range(n)]

def mat_val(n: int, m: int, v: Any) -> List[List[Any]]:
    # Crée une matrice de taille (n × m) remplie avec la valeur v
    return [vec_val(m, v) for _ in range(n)]

def mat_transpose(X: List[List[float]]) -> List[List[float]]:
    # Transposition d’une matrice : transforme X (n × m) en (m × n)
    return [list(row) for row in zip(*X)]

def mat_sum(X: List[List[float]], col: bool = False) -> List[float]:
    # Somme des lignes (par défaut) ou des colonnes (si col=True) d’une matrice
    X2 = zip(*X) if col else X
    return [sum(V) for V in X2]

def mat_mean(X: List[List[float]], col: bool = False) -> List[float]:
    # Moyenne des lignes (par défaut) ou des colonnes (si col=True) d’une matrice
    if col:
        X2, l = zip(*X), len(X)
    else:
        X2, l = X, len(X[0])
    return [sum(V) / l for V in X2]

def mat_mean_var(X: List[List[float]], col: bool = False) -> Tuple[List[float], List[float]]:
    # Moyenne et variance des lignes (par défaut) ou des colonnes (si col=True) d’une matrice
    if col:
        X2, l = zip(*X), len(X)
    else:
        X2, l = X, len(X[0])
    
    means = [sum(V) / l for V in X2]
    vars_ = [sum((x - mean)**2 for x in V) / l for V, mean in zip(X2, means)]
    
    return means, vars_

def mat_mean_std(X: List[List[float]], col: bool = False, eps: float = 1e-5) -> Tuple[List[float], List[float]]:
    # Moyenne et écart type des lignes (par défaut) ou des colonnes (si col=True), avec epsilon pour la stabilité numérique
    if col:
        X2, l = zip(*X), len(X)
    else:
        X2, l = X, len(X[0])

    means = [sum(V) / l for V in X2]
    stds = [math.sqrt(sum((x - mean)**2 for x in V) / l + eps) for V, mean in zip(X2, means)]

    return means, stds

def mat_normalize(X: List[List[float]], mu: List[float], std: List[float], col: bool = False, eps: float = 1e-5) -> List[List[float]]:
    # Normalisation d’une matrice selon les moyennes et écarts types donnés (lignes par défaut, colonnes si col=True)
    if col:
        X = list(zip(*X))  

    X_norm = [[(x - m) / (s + eps) for x, m, s in zip(row, mu, std)] for row in X]

    if col:
        X_norm = list(map(list, zip(*X_norm))) 

    return X_norm

def mat_smul(X: List[List[float]], s: float) -> List[List[float]]:
    # Multiplication d’une matrice par un scalaire : X * s
    return list(map(lambda x: vec_smul(x, s), X))

def mat_softmax(X: List[List[float]], M: List[List[bool]]=None, col: bool = False) -> List[List[float]]:
    # Application de la fonction softmax sur chaque ligne (par défaut) ou chaque colonne (si col=True) d’une matrice
    if col:
        X = list(zip(*X))  
    if M is not None:
        return list(map(lambda x_m: vec_softmax(x_m[0], x_m[1]), X, M))
    
    return list(map(lambda x: vec_softmax(x), X))