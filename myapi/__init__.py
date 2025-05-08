class Module:
    def forward(self, X):
        # Passage avant : doit retourner la sortie
        raise NotImplementedError

    def backward(self, dJ):
        # Passage arrière : doit retourner le gradient par rapport à l'entrée
        raise NotImplementedError

class Layer(Module):
    pass