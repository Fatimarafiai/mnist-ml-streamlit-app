# Machine Learning sur MNIST : application Streamlit interactive

> Boîte à outils interactive pour explorer, comparer et comprendre des méthodes de réduction de dimension et de classification sur le dataset MNIST, avec la théorie mathématique et une visualisation « avant / après » pour chaque méthode.

Projet académique réalisé dans le cadre du Master Intelligence Artificielle pour l'Économie Numérique et la Gestion (Faculté des Sciences Juridiques, Économiques et Sociales de Tanger, Université Abdelmalek Essaâdi).

## Fonctionnalités

L'application regroupe **10 techniques** accessibles depuis la barre latérale. Pour chacune : rappel théorique et formules, code correspondant, hyperparamètres réglables en direct, métriques et comparaison avant / après.

| Famille | Méthodes |
|---|---|
| Réduction de dimension | PCA, LDA, t-SNE, UMAP |
| Classification | QDA, SVM, Decision Tree, KNN |
| Détection d'anomalies | One-Class SVM |
| Traitement du signal | Compressive Sensing |

### Ce que montre chaque section

- **PCA** : compression et reconstruction d'images avec *k* composantes, variance expliquée, projection 2D, matrice de confusion d'un KNN entraîné sur la projection.
- **LDA** : approches de Fisher et bayésienne, comparaison avec et sans *shrinkage*, comparaison PCA (non supervisée) vs LDA (supervisée).
- **QDA** : frontière quadratique après pré-réduction par PCA, comparaison avec la baseline linéaire LDA, effet de `reg_param`.
- **t-SNE** et **UMAP** : projection non linéaire vs PCA, réglage de `perplexity`, `n_neighbors` et `min_dist`, évaluation de la qualité de la projection par matrice de confusion.
- **SVM** : choix du noyau (`rbf`, `linear`, `poly`, `sigmoid`), nombre de vecteurs de support, visualisation de la frontière de décision en 2D, démonstration du *kernel trick*.
- **One-Class SVM** : apprentissage sur un seul chiffre « normal » puis détection des anomalies.
- **Decision Tree** : visualisation de l'arbre, importance des pixels, critères `gini` / `entropy` / `log_loss`.
- **KNN** : vote des *k* plus proches voisins, visualisation du voisinage, courbe d'accuracy en fonction de *k*.
- **Compressive Sensing** : reconstruction d'images depuis des mesures compressées, comparaison L2 (naïve) et L1 (Lasso dans la base DCT).

## Données

Deux versions de MNIST au choix dans la barre latérale :

| Version | Taille | Source |
|---|---|---|
| Digits | 8×8 pixels | `sklearn.datasets.load_digits` (hors ligne, rapide) |
| MNIST complet | 28×28 pixels | OpenML (`mnist_784`), sous-échantillon de 500 à 10 000 images |

Aucun fichier de données n'est à télécharger manuellement : les datasets sont chargés automatiquement et mis en cache.

## Technologies

Python, Streamlit, scikit-learn, UMAP, NumPy, pandas, SciPy, Matplotlib, Seaborn.

## Installation et lancement

```bash
# 1. Cloner le dépôt
git clone https://github.com/Fatimarafiai/mnist-ml-streamlit-app.git
cd mnist-ml-streamlit-app

# 2. (Recommandé) créer un environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`.

> Prérequis : Python 3.9 ou plus, scikit-learn 1.5 ou plus. Le premier chargement de « MNIST complet » nécessite une connexion internet.

## Aperçu

### Projection PCA (MNIST complet, 6 000 images)

![Projection PCA](Screenshot%202026-10-24%20225209.png)

### SVM : vraies classes vs prédictions (erreurs entourées en rouge)

![SVM avant / après](Screenshot%202026-10-24%20225029.png)

## Structure du dépôt

- `app.py` : application Streamlit
- `requirements.txt` : dépendances Python
- `Screenshot 2026-10-24 225209.png`, `Screenshot 2026-10-24 225029.png` : captures d'écran de l'application
- `README.md` : ce fichier

## Limites

- Les graphiques en 2D (frontière du SVM, cercle de voisinage du KNN) utilisent une projection PCA à but pédagogique : ils ne représentent pas exactement l'espace de travail réel du modèle, ce qui est précisé dans l'application.
- Sur de petits échantillons, certaines méthodes (LDA, QDA) sont sensibles au nombre d'images par classe ; l'application propose le *shrinkage* et la régularisation pour y remédier.

## Contact

Fatima Rafiai · fatimarafiai0101@gmail.com · Tanger, Maroc
