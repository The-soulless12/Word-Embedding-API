import math
from typing import List, Tuple
from myapi.mat_func import mat_add_vec, mat_dot, mat_ew_op, mat_random, mat_transpose, mat_ew_sub, mat_smul
from . import Layer
from myapi.vec_func import vec_ew_mul, vec_ew_op, vec_random, vec_sum, vec_val, vec_smul, vec_softmax, vec_ew_sub

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
        self.eps = eps
        self.gamma = vec_val(dim, 1.0)  # shape: (features,)
        self.beta = vec_val(dim, 0.0)   # shape: (features,)

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        # Applique la normalisation couche par couche sur le dernier axe (features)
        self.X = X
        self.mean, self.std, self.norm = [], [], []
        O = []

        for x_batch in X:
            mean_batch, std_batch, norm_batch, o_batch = [], [], [], []

            for x in x_batch:
                mu = vec_sum(x) / len(x)
                variance = vec_sum([(xi - mu) ** 2 for xi in x]) / len(x)
                std = (variance + self.eps) ** 0.5

                x_norm = [(xi - mu) / std for xi in x]
                out = [self.gamma[i] * x_norm[i] + self.beta[i] for i in range(len(x))]

                mean_batch.append(mu)
                std_batch.append(std)
                norm_batch.append(x_norm)
                o_batch.append(out)

            self.mean.append(mean_batch)
            self.std.append(std_batch)
            self.norm.append(norm_batch)
            O.append(o_batch)

        return O

    def backward(self, dY: List[List[List[float]]], alpha: float = 0.01) -> List[List[List[float]]]:
        # Calcule le gradient par rapport à l’entrée, gamma et beta et met à jour ces derniers
        dgamma = vec_val(len(self.gamma), 0.0)
        dbeta = vec_val(len(self.beta), 0.0)

        # Première passe : calcul des gradients de gamma et beta
        for dy_batch, norm_batch in zip(dY, self.norm):
            for dy, x_norm in zip(dy_batch, norm_batch):
                for i in range(len(dy)):
                    dgamma[i] += dy[i] * x_norm[i]
                    dbeta[i] += dy[i]

        # Mise à jour des paramètres gamma et beta avec le taux d’apprentissage alpha
        for i in range(len(self.gamma)):
            self.gamma[i] -= alpha * dgamma[i]
            self.beta[i] -= alpha * dbeta[i]

        # Deuxième passe : calcul du gradient par rapport à l’entrée X
        dX = []
        for x_batch, dy_batch, mean_batch, std_batch in zip(self.X, dY, self.mean, self.std):
            dX_batch = []

            for x, dy, mu, std in zip(x_batch, dy_batch, mean_batch, std_batch):
                dx_norm = vec_ew_mul(dy, self.gamma)

                dvar = vec_sum([
                    dx_norm[j] * (x[j] - mu) * -0.5 / (std ** 3)
                    for j in range(len(x))
                ])

                dmean = vec_sum([
                    -dx_norm[j] / std for j in range(len(x))
                ]) + dvar * vec_sum([
                    -2 * (x[j] - mu) for j in range(len(x))
                ]) / len(x)

                dx = [
                    dx_norm[j] / std +
                    dvar * 2 * (x[j] - mu) / len(x) +
                    dmean / len(x)
                    for j in range(len(x))
                ]
                dX_batch.append(dx)

            dX.append(dX_batch)

        return dX

class ScaledDotProductAttention(Layer):
    def __init__(self) -> None:
        # Pas de paramètres à apprendre dans cette couche
        pass 

    def forward_single(self, Q: List[List[float]], K: List[List[float]], V: List[List[float]], M: List[List[bool]] = None) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        # Prédit la sortie pour une seule entrée (T, d)
        d = len(Q[0])
        KT = mat_transpose(K)
        S = mat_dot(Q, KT)
        S = mat_smul(S, math.sqrt(d))  

        if M:
            for i in range(len(M)):
                for j in range(len(M[i])):
                    if not M[i][j]:
                        S[i][j] = -float('inf')

        P = [vec_softmax(s_row) for s_row in S]
        Y = mat_dot(P, V)

        return Y, S, P

    def forward(self, Qs: List[List[List[float]]], Ks: List[List[List[float]]], Vs: List[List[List[float]]], Ms: List[List[List[bool]]] = None) -> List[List[List[float]]]:
        # Prédit la sortie pour un lot d'entrées (M, T, d)
        self.Qs, self.Ks, self.Vs, self.Ms = Qs, Ks, Vs, Ms
        self.Ss = []
        self.Ps = []
        Ys      = []

        for i in range(len(Qs)):
            Q, K, V, M = Qs[i], Ks[i], Vs[i], Ms[i] if Ms else None
            Y, S, P = self.forward_single(Q, K, V, M)
            
            self.Ss.append(S)
            self.Ps.append(P)
            Ys.append(Y)

        return Ys

    def backward_single(self, dY: List[List[float]], alpha: float = 0.01) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        # Rétropropagation pour une seule entrée (T, d)
        dP = mat_dot(dY, mat_transpose(self.Vs))  
        dV = mat_dot(mat_transpose(self.Ps), dY)  
        
        dS = mat_dot(dY, self.Ks)
        dK = mat_dot(mat_transpose(self.Qs), dS)  
        dQ = mat_dot(mat_transpose(self.Vs), dS) 

        dQ = vec_smul(dQ, alpha)
        dK = vec_smul(dK, alpha)
        dV = vec_smul(dV, alpha)
        return dQ, dK, dV

    def backward(self, dYs: List[List[List[float]]], alpha: float = 0.01) -> Tuple[List[List[List[float]]], List[List[List[float]]], List[List[List[float]]]]:
        # Rétropropagation pour un lot d'entrées (M, T, d)
        dQs = []
        dKs = []
        dVs = []

        for i in range(len(dYs)):
            dQ, dK, dV = self.backward_single(dYs[i], alpha)
            dQs.append(dQ)
            dKs.append(dK)
            dVs.append(dV)

        return dQs, dKs, dVs