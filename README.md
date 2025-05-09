# Word-Embedding-API
Framework de Deep Learning en Python intégrant des mécanismes de Word Embedding pour la représentation vectorielle des mots, facilitant l'entraînement de modèles classiques.

# Fonctionnalités
- Conversion de mots en vecteurs denses et continus pour capturer la sémantique et les relations entre les mots à l'aide du module d'Embedding.
- Implémentation de mécanismes d'attention (Scaled Dot-Product Attention) permettant de pondérer l'importance des différentes parties des données.
- Stabilisation de l'apprentissage par la normalisation des activations au sein des couches du réseau (Layer Normalization).

# Structure du Projet
- myapi/ : Contient le code source du framework.
  - _\_init_\_.py : Définit les classes de base Module et Layer.
  - vec_func.py : Contient l'implémentation des fonctions vectorielles (addition, multiplication, normalisation, etc.).
  - mat_func.py : Contient l'implémentation des fonctions matricielles (produit matriciel, transposition, etc.).
  - layers_starter.py : Implémentation des couches spécifiques (Linear, Embedding, LayerNorm & ScaledDotProductAttention).
- test/ : Contient les tests unitaires pour valider les fonctionnalités.
  - embedding_u.py : Tests de la couche d'Embedding.
  - norm_u.py : Tests de la couche de normalisation.
  - attention_u.py : Tests du mécanisme d'attention.

## Prérequis
- Python 3.7+
- Le package : pytest.

## Note
- Pour exécuter le projet et tester la normalisation de couche, saisissez la commande `pytest test/norm_u.py`, pour tester l'embedding saisissez `pytest test/embedding_u.py` et pour tester le mécanisme d'attention saisissez `pytest test/attention_u.py` dans votre terminal.
- Ce projet est une implémentation pure Python des mécanismes fondamentaux du NLP moderne, conçue pour comprendre les principes des word embeddings et de l'attention **sans dépendances externes aux grandes bibliothèques de deep learning**.