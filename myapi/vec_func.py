import math
import random
from functools import reduce
from typing import List, Any

def vec_ew_add(X: List[float], Y: List[float]) -> List[float]:
    # Addition élément par élément de deux vecteurs X et Y
    return list(map(sum, zip(X, Y)))

def vec_ew_sub(X: List[float], Y: List[float]) -> List[float]:
    # Soustraction élément par élément de deux vecteurs X et Y
    return list(map(lambda x_y: x_y[0] - x_y[1], zip(X, Y)))

def vec_ew_mul(X: List[float], Y: List[float]) -> List[float]:
    # Multiplication élément par élément de deux vecteurs X et Y
    return list(map(lambda x_y: x_y[0] * x_y[1], zip(X, Y)))

def vec_ew_op(X: List[float], Y: List[float], op) -> List[float]:
    # Applique une opération binaire élément par élément sur deux vecteurs X et Y
    return list(map(lambda x_y: op(x_y[0], x_y[1]), zip(X, Y)))

def vec_dot(X: List[float], Y: List[float]) -> float:
    # Calcul du produit scalaire entre deux vecteurs X et Y
    return reduce(lambda acc, x_y: acc + x_y[0] * x_y[1], zip(X, Y), 0.0)

def vec_norm(X: List[float]) -> float:
    # Calcul de la norme L2 du vecteur X
    return math.sqrt(reduce(lambda acc, x: acc + x*x, X, 0.0))

def vec_exp(X: List[float]) -> List[float]:
    # Applique l'exponentielle à chaque élément du vecteur X
    return list(map(math.exp, X))

def vec_smul(X: List[float], s: float) -> List[float]:
    # Multiplie chaque élément du vecteur X par un scalaire s
    return list(map(lambda e: e * s, X))

def vec_shift(X: List[float], v: float) -> List[float]:
    # Décale chaque élément du vecteur X par une valeur v
    return list(map(lambda x: x - v, X))

def vec_sum(X: List[float]) -> float:
    # Somme de tous les éléments du vecteur X
    return reduce(lambda acc, x: acc + x, X, 0.0)

def vec_softmax(X: List[float], M: List[float]=None) -> List[float]:
    # Applique la fonction softmax à un vecteur X avec un masque M (facultatif)
    r = vec_exp(vec_shift(X, max(X)))
    if M is not None:
        r = vec_ew_mul(r, M)
    s = vec_sum(r)
    if s == 0.0:
        return [0.0] * len(X) 

    return vec_smul(r, 1/s)

def vec_concat(*vectors: List[float]) -> List[float]:
    # Concatène plusieurs vecteurs en un seul
    return reduce(lambda acc, X: acc + X, vectors, [])

def vec_random(n: int) -> List[float]:
    # Crée un vecteur de taille n avec des valeurs aléatoires
    return [random.random() for _ in range(n)]

def vec_val(n: int, v: Any) -> List[Any]:
    # Crée un vecteur de taille n rempli avec la valeur v
    return [v] * n

def onehot(lst: List[Any], e: Any) -> List[int]:
    # Encode un élément en one-hot selon une liste donnée
    res = [0] * len(lst)
    if e in lst:
        res[lst.index(e)] = 1
    return res

def get_sentences(url: str) -> List[List[str]]:
    # Récupère les phrases à partir d'un fichier URL et les retourne sous forme de liste de mots
    f = open(url, 'r', encoding='utf8')
    data: List[List[str]] = []
    for l in f: 
        if len(l) < 5: 
            continue
        data.append(l.strip(' \t\r\n').split())
            
    f.close()
    return data