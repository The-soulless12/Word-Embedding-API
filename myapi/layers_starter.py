import math
from typing import List, Tuple
from myapi.mat_func import mat_add_vec, mat_dot, mat_ew_op, mat_random, mat_transpose, mat_ew_sub, mat_smul, mat_ew_mul, mat_softmax
from . import Layer
from myapi.vec_func import vec_ew_mul, vec_ew_op, vec_random, vec_sum, vec_val, vec_smul, vec_ew_sub, vec_dot, vec_ew_add

class Linear(Layer):
    def __init__(self, in_size: int, out_size: int, bias: bool = False) -> None:
        # Initialise une couche linéaire avec des poids et un biais (optionnel).
        self.w = mat_random(out_size, in_size)
        self.bias = bias
        if bias:
            self.b = vec_random(out_size)
    
    def forward_single(self, X: List[float]) -> List[float]:
        # Prédit la sortie pour une seule entrée en utilisant la méthode forward.
        return self.forward([X])[0]
    
    def forward(self, Xs: List[List[float]]) -> List[List[float]]:
        # Prédit la sortie pour plusieurs entrées en multipliant les entrées par les poids.
        self.Xs = Xs 
        Ys = mat_dot(Xs, self.w)
        if self.bias:
            Ys = mat_add_vec(Ys, self.b)
        return Ys 

    def backward_single(self, dY: List[float], alpha:float=0.01) -> None:
        # Effectue la rétropropagation et met à jour les poids pour une seule entrée.
        return self.backward([dY], alpha=alpha)
    
    def backward(self, dYs: List[List[float]], alpha:float=0.01) -> None:
        # Effectue la rétropropagation et met à jour les poids pour plusieurs entrées.
        dW =  mat_dot(mat_transpose(dYs), self.X)

        self.w = mat_ew_op(self.w, dW, lambda w, g: w - alpha * g)

        if self.bias:
            self.w = vec_ew_op(self.w, mat_transpose(dW), lambda w, G: w - alpha * vec_sum(G))

        return mat_dot(dYs, self.w)

class Embedding(Layer):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        # Initialise la matrice d'embeddings
        self.w = mat_random(vocab_size, embed_dim)
        
    def forward_single(self, X: List[int]) -> List[List[float]]:
        # Passe avant pour une seule séquence
        self.indices = X  
        return [self.w[token] for token in X]

    def forward(self, Xs: List[List[int]]) -> List[List[List[float]]]:
        # Passe avant pour un lot de séquences
        self.batch_indices = Xs 
        return [self.forward_single(seq) for seq in Xs]

    def backward_single(self, dY: List[List[float]], indices: List[int], alpha: float = 0.01) -> None:
        # Rétropropagation pour une seule séquence
        for i, token_idx in enumerate(indices):
            self.w[token_idx] = vec_ew_sub(self.w[token_idx], vec_smul(dY[i], alpha))

    def backward(self, dYs: List[List[List[float]]], alpha: float = 0.01) -> None:
        # Rétropropagation pour un lot de séquences
        for batch_idx, dY in enumerate(dYs):
            self.backward_single(dY, self.batch_indices[batch_idx], alpha)

class LayerNorm(Layer):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        # Initialise la normalisation de couche
        self.eps = eps
        self.gamma = vec_val(dim, 1.0)
        self.beta = vec_val(dim, 0.0)

    def forward(self, X: list[list[list[float]]]) -> list[list[list[float]]]:
        # Applique la normalisation couche par couche sur le dernier axe (features)
        self.X = X
        batch_means: list[list[float]] = []
        batch_stds:  list[list[float]] = []
        batch_norms: list[list[list[float]]] = []
        output:     list[list[list[float]]] = []

        for batch in X:
            means: list[float] = []
            stds:  list[float] = []
            norms: list[list[float]] = []
            outs:  list[list[float]] = []

            for x in batch:
                n = len(x)
                mu  = sum(x) / n
                var = sum((xi - mu) ** 2 for xi in x) / n
                std = math.sqrt(var + self.eps)
                denom = std + self.eps

                xn = [(xi - mu) / denom for xi in x]
                xo = vec_ew_add(vec_ew_mul(xn, self.gamma), self.beta)

                means.append(mu)
                stds.append(std)
                norms.append(xn)
                outs.append(xo)

            batch_means.append(means)
            batch_stds.append(stds)
            batch_norms.append(norms)
            output.append(outs)

        self.mean = batch_means
        self.std  = batch_stds
        self.norm = batch_norms
        
        return output

    def backward(self, dY: list[list[list[float]]], alpha: float = 0.1) -> list[list[list[float]]]:
        # Rétropropagation de la normalisation de couche
        dgamma = vec_val(len(self.gamma), 0.0)
        dbeta  = vec_val(len(self.beta),  0.0)
        for dy_batch, xn_batch in zip(dY, self.norm):
            for dy, xn in zip(dy_batch, xn_batch):
                dgamma = vec_ew_add(dgamma, vec_ew_mul(dy, xn))
                dbeta  = vec_ew_add(dbeta, dy)
        self.gamma = vec_ew_op(self.gamma, dgamma, lambda g, dg: g - alpha * dg)
        self.beta  = vec_ew_op(self.beta,  dbeta, lambda b, db: b - alpha * db)

        dX: list[list[list[float]]] = []
        for x_batch, dy_batch, mean_batch, std_batch in zip(self.X, dY, self.mean, self.std):
            dX_b: list[list[float]] = []
            for x, dy, mu, std in zip(x_batch, dy_batch, mean_batch, std_batch):
                dxn = vec_ew_mul(dy, self.gamma)
                dvar = sum(dxn[j] * (x[j] - mu) * -0.5 / (std**3) for j in range(len(x)))
                dmean = (
                    sum(-dxn[j] / std for j in range(len(x)))
                    + dvar * sum(-2 * (x[j] - mu) for j in range(len(x))) / len(x)
                )
                dx = [
                    dxn[j] / std
                    + dvar * 2 * (x[j] - mu) / len(x)
                    + dmean / len(x)
                    for j in range(len(x))
                ]
                dX_b.append(dx)
            dX.append(dX_b)

        return dX

class ScaledDotProductAttention(Layer):
    def __init__(self) -> None:
        # Pas de paramètres à apprendre dans cette couche
        pass 

    def forward_single(self, Q: List[List[float]], K: List[List[float]], V: List[List[float]], M: List[List[bool]] = None) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        # Calcule l'attention pour une seule entrée
        d = len(Q[0])
        scale = 1 / math.sqrt(d)

        S = mat_smul(mat_dot(Q, mat_transpose(K)), scale)
        P = mat_softmax(S, M)
        Y = mat_dot(P, V)

        return S, P, Y

    def forward(self, Qs: List[List[List[float]]], Ks: List[List[List[float]]], Vs: List[List[List[float]]], Ms: List[List[List[bool]]] = None) -> List[List[List[float]]]:
        # Calcule l'attention pour plusieurs entrées
        self.Qs, self.Ks, self.Vs, self.Ms = Qs, Ks, Vs, Ms
        self.Ss, self.Ps, Ys = [], [], []

        for i in range(len(Qs)):
            S, P, Y = self.forward_single(Qs[i], Ks[i], Vs[i], Ms[i] if Ms else None)
            self.Ss.append(S)
            self.Ps.append(P)
            Ys.append(Y)

        return Ys

    def backward_single(self, dY: List[List[float]], Q: List[List[float]], K: List[List[float]], V: List[List[float]], P: List[List[float]], S: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        # Rétropropagation pour une seule entrée
        d = len(Q[0])
        scale = 1 / math.sqrt(d)

        dP = mat_dot(dY, mat_transpose(V))        
        dS = mat_ew_mul(P, mat_ew_sub(dP, [[vec_dot(dP[i], P[i]) for _ in range(len(P[i]))] for i in range(len(P))]))
        dQ = mat_smul(mat_dot(dS, K), scale)
        dK = mat_smul(mat_dot(mat_transpose(dS), Q), scale)
        dV = mat_dot(mat_transpose(P), dY)

        return dQ, dK, dV

    def backward(self, dYs: List[List[List[float]]], alpha: float = 0.01) -> Tuple[List[List[List[float]]], List[List[List[float]]], List[List[List[float]]]]:
        # Rétropropagation pour plusieurs entrées
        dQs, dKs, dVs = [], [], []
        for dY, Q, K, V, P, S in zip(dYs, self.Qs, self.Ks, self.Vs, self.Ps, self.Ss):
            dQ, dK, dV = self.backward_single(dY, Q, K, V, P, S)
            dQs.append(dQ)
            dKs.append(dK)
            dVs.append(dV)
        return dQs, dKs, dVs