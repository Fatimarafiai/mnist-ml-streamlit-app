"""
Application Streamlit — Boîte à outils Machine Learning sur MNIST
====================================================================
Regroupe : PCA, LDA, QDA, t-SNE, UMAP, SVM, One-Class SVM,
Decision Tree, KNN — avec théorie mathématique et comparaison
avant / après pour chaque méthode.

Lancer avec :  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.manifold import TSNE
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import Lasso
from scipy.fftpack import dct, idct

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ----------------------------------------------------------------------
# Configuration générale de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="ML sur MNIST — Théorie & Pratique",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔢 Machine Learning sur MNIST — Théorie & Pratique")
st.caption(
    "PCA · LDA · QDA · t-SNE · UMAP · SVM · One-Class SVM · "
    "Decision Tree · KNN — avec maths et visualisation avant/après"
)


# ----------------------------------------------------------------------
# 1. Chargement des données — MNIST uniquement
# ----------------------------------------------------------------------
st.sidebar.header("📁 Données — MNIST")

mnist_variant = st.sidebar.radio(
    "Version de MNIST",
    ["Digits (8×8, rapide, hors-ligne)", "MNIST complet (28×28, via OpenML)"],
)


@st.cache_data(show_spinner=False)
def load_digits_dataset():
    raw = load_digits(as_frame=True)
    return raw.frame, (8, 8)


@st.cache_data(show_spinner=False)
def load_full_mnist(n_samples):
    mnist = fetch_openml("mnist_784", version=1, as_frame=True, parser="auto")
    X = mnist.data
    y = mnist.target.astype(int)
    if n_samples < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    frame = X.copy()
    frame["target"] = y.values
    return frame, (28, 28)


df, image_shape = None, None
target_col = "target"

if mnist_variant.startswith("Digits"):
    with st.spinner("Chargement de Digits..."):
        df, image_shape = load_digits_dataset()

else:
    st.sidebar.warning(
        "⚠️ Téléchargement depuis OpenML requis. Peut prendre du temps "
        "la première fois (mis en cache ensuite)."
    )
    n_samples = st.sidebar.slider(
        "Nombre d'échantillons à charger", 500, 10000, 3000, step=500
    )
    if st.sidebar.button("📥 Charger MNIST complet"):
        with st.spinner(f"Chargement de {n_samples} images MNIST..."):
            df, image_shape = load_full_mnist(n_samples)
        st.session_state["mnist_df"] = df
        st.session_state["mnist_shape"] = image_shape
    elif "mnist_df" in st.session_state:
        df = st.session_state["mnist_df"]
        image_shape = st.session_state["mnist_shape"]

if df is None:
    st.info("⬅️ Charge une version de MNIST dans la barre latérale pour commencer.")
    st.stop()

pixel_cols = [c for c in df.columns if c != target_col]

with st.expander("👀 Aperçu des données", expanded=False):
    st.write(f"Dimensions : {df.shape[0]} images × {len(pixel_cols)} pixels")
    n_preview = 10
    cols = st.columns(n_preview)
    for i in range(n_preview):
        img = df.iloc[i][pixel_cols].values.astype(float).reshape(image_shape)
        with cols[i]:
            fig, ax = plt.subplots(figsize=(1.2, 1.2))
            ax.imshow(img, cmap="gray")
            ax.set_title(str(df.iloc[i][target_col]), fontsize=9)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)


# ----------------------------------------------------------------------
# 2. Préparation X / y
# ----------------------------------------------------------------------
X_raw = df[pixel_cols].astype(float)
y_raw = df[target_col]
le = LabelEncoder()
y = le.fit_transform(y_raw.astype(str))
class_names = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

st.sidebar.divider()
st.sidebar.header("⚙️ Technique")

section = st.sidebar.radio(
    "Choisir une méthode",
    [
        "PCA", "LDA", "QDA", "t-SNE", "UMAP",
        "SVM", "One-Class SVM", "Decision Tree", "KNN",
        "Compressive Sensing",
    ],
)


# ----------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------
def plot_2d_projection(coords, y, class_names, title, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10", s=18, alpha=0.8)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, class_names, title="Chiffre", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    ax.set_title(title)
    if standalone:
        st.pyplot(fig)
        plt.close(fig)


def show_classification_results(y_test, y_pred, class_names, model_name):
    acc = accuracy_score(y_test, y_pred)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(f"Accuracy — {model_name}", f"{acc:.2%}")
        report = classification_report(
            y_test, y_pred, target_names=[str(c) for c in class_names],
            output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3))
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)


def train_test_split_data(X, y, test_size=0.3, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def before_after_classification(X_train, X_test, y_test, y_pred, class_names, model_name):
    """Avant = vraies classes (projection PCA) — Après = classes prédites, erreurs entourées en rouge."""
    pca_viz = PCA(n_components=2, random_state=42).fit(X_train)
    coords_test = pca_viz.transform(X_test)
    errors = y_pred != y_test

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔵 Avant — vraies classes (vérité terrain)**")
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_2d_projection(coords_test, y_test, class_names, "Vérité terrain (projection PCA)", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown(f"**🟢 Après — prédictions {model_name} (erreurs entourées)**")
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(coords_test[:, 0], coords_test[:, 1], c=y_pred, cmap="tab10", s=18, alpha=0.8)
        ax.scatter(
            coords_test[errors, 0], coords_test[errors, 1],
            facecolors="none", edgecolors="red", s=90, linewidths=1.5, label="Erreur"
        )
        handles, _ = scatter.legend_elements()
        ax.legend(handles, class_names, title="Prédit", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.set_xlabel("Composante 1")
        ax.set_ylabel("Composante 2")
        ax.set_title(f"{model_name} — {int(errors.sum())} erreurs / {len(y_test)}")
        st.pyplot(fig)
        plt.close(fig)


def embedding_confusion_matrix(coords, y, class_names, k_neighbors=5):
    """
    Entraîne un petit classifieur KNN directement sur les coordonnées de la
    projection (PCA / t-SNE / UMAP) pour évaluer, via une matrice de confusion,
    à quel point cette projection sépare bien les classes.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        coords, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = KNeighborsClassifier(n_neighbors=k_neighbors)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Accuracy d'un KNN entraîné sur cette projection", f"{acc:.2%}")
        st.caption(
            "Plus cette accuracy est élevée, plus la projection 2D sépare bien "
            "les classes (même si elle n'a jamais utilisé les labels pour se "
            "construire, dans le cas de PCA/t-SNE/UMAP)."
        )
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_te, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)
    return acc


def dct2(a):
    """Transformée en cosinus discrète 2D (base de parcimonie, comme en JPEG)."""
    return dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(a):
    """Transformée en cosinus discrète inverse 2D."""
    return idct(idct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


@st.cache_data(show_spinner=False)
def build_idct_basis(shape):
    """Construit la matrice Psi (base IDCT) : Psi @ s = image reconstruite depuis ses coefficients DCT s."""
    n = shape[0] * shape[1]
    Psi = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        Psi[:, i] = idct2(e.reshape(shape)).flatten()
    return Psi


# ----------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------
if section == "PCA":
    st.header("📉 PCA — Analyse en Composantes Principales")

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "La PCA a été inventée par **Karl Pearson en 1901** (\"On Lines and "
            "Planes of Closest Fit to Systems of Points in Space\"), puis "
            "formalisée sous sa forme statistique moderne par **Harold Hotelling "
            "en 1933**. C'est l'une des plus anciennes méthodes de réduction de "
            "dimension, encore aujourd'hui la plus utilisée."
        )
        st.markdown(
            "La PCA est une méthode **non supervisée** qui cherche les directions "
            "de variance maximale dans les données, pour les projeter dans un "
            "espace de dimension réduite en perdant le moins d'information possible."
        )
        st.markdown("**1. Centrer les données**")
        st.latex(r"X_c = X - \bar{X}")
        st.markdown("**2. Calculer la matrice de covariance**")
        st.latex(r"\Sigma = \frac{1}{n} X_c^\top X_c")
        st.markdown("**3. Diagonaliser** (vecteurs et valeurs propres)")
        st.latex(r"\Sigma v_i = \lambda_i v_i")
        st.markdown(
            "Les vecteurs propres $v_i$ (composantes principales) sont triés par "
            "valeur propre $\\lambda_i$ décroissante (= variance expliquée)."
        )
        st.markdown("**4. Projection sur les $k$ premières composantes**")
        st.latex(r"Z = X_c V_k")
        st.markdown("**5. Reconstruction approximative (compression)**")
        st.latex(r"\hat{X} = Z V_k^\top + \bar{X}")
        st.markdown(
            "Sur MNIST (784 pixels pour l'image 28×28), la PCA permet de résumer "
            "chaque image avec seulement quelques dizaines de nombres au lieu de "
            "784, tout en conservant l'essentiel de l'information visuelle."
        )

    with st.expander("💻 Le code correspondant", expanded=False):
        st.markdown("**Reconstruction d'image (compression avec $k$ composantes)**")
        st.code(
            '''from sklearn.decomposition import PCA

# X_pixels : matrice (n_images, 784) des pixels bruts
pca_recon = PCA(n_components=k_recon)     # k_recon = nb de composantes gardées
Z = pca_recon.fit_transform(X_pixels)     # projection : (n_images, k_recon)
X_hat = pca_recon.inverse_transform(Z)    # reconstruction : (n_images, 784)

# Variance conservée avec ces k_recon composantes :
variance_kept = pca_recon.explained_variance_ratio_.sum()''',
            language="python",
        )
        st.markdown(
            "`fit_transform` calcule les axes principaux puis projette chaque "
            "image sur les `k_recon` premiers. `inverse_transform` fait le chemin "
            "inverse (projection → pixels) : c'est une reconstruction *approximative*, "
            "d'où la perte de détails visible sur l'image \"après\"."
        )
        st.markdown("**Projection 2D et variance expliquée**")
        st.code(
            '''pca = PCA(n_components=n_components)
coords = pca.fit_transform(X_scaled)          # X_scaled = pixels centrés-réduits
var_ratio = pca.explained_variance_ratio_     # % de variance par axe
# coords[:, 0] = axe 1 (abscisse du graphique)
# coords[:, 1] = axe 2 (ordonnée du graphique)''',
            language="python",
        )
        st.markdown(
            "Chaque **axe du graphique** est une composante principale : une "
            "combinaison linéaire des 784 pixels d'origine, choisie pour capter "
            "un maximum de variance. L'axe 1 (horizontal) capte toujours plus de "
            "variance que l'axe 2 (vertical), etc."
        )

    st.subheader("🖼️ Avant / Après — Compression d'image par PCA")
    idx = st.slider("Choisir une image", 0, len(X_raw) - 1, 0)
    k_recon = st.slider("Nombre de composantes pour la reconstruction", 1, min(150, X_raw.shape[1]), 20)

    X_pixels = X_raw.values
    pca_recon = PCA(n_components=k_recon, random_state=42)
    Z = pca_recon.fit_transform(X_pixels)
    X_hat = pca_recon.inverse_transform(Z)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔵 Avant — image originale (784 pixels)**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(X_pixels[idx].reshape(image_shape), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Chiffre {y_raw.iloc[idx]}")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.markdown(f"**🟢 Après — reconstruite avec {k_recon} composantes**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(X_hat[idx].reshape(image_shape), cmap="gray")
        ax.axis("off")
        variance_kept = pca_recon.explained_variance_ratio_.sum()
        ax.set_title(f"{variance_kept:.1%} de variance conservée")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("📊 Projection 2D et variance expliquée")
    n_components = st.slider("Nombre de composantes (visualisation)", 2, min(10, X_scaled.shape[1]), 2)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    col_ax1, col_ax2 = st.columns(2)
    with col_ax1:
        axis_x = st.selectbox("Axe X", list(range(1, n_components + 1)), index=0, key="pca_axis_x")
    with col_ax2:
        axis_y = st.selectbox("Axe Y", list(range(1, n_components + 1)), index=min(1, n_components - 1), key="pca_axis_y")

    plot_2d_projection(
        coords[:, [axis_x - 1, axis_y - 1]], y, class_names,
        f"Projection PCA (composante {axis_x} vs composante {axis_y})"
    )
    st.caption(
        "💡 Augmente le nombre de composantes ci-dessus puis change les axes "
        "X/Y pour explorer d'autres combinaisons — les 2 premières composantes "
        "captent le plus de variance, mais pas forcément la meilleure séparation "
        "visuelle entre certains chiffres précis."
    )

    var_ratio = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(1, n_components + 1), var_ratio, color="steelblue")
    ax.plot(range(1, n_components + 1), np.cumsum(var_ratio), color="orange", marker="o", label="Cumulée")
    ax.set_xlabel("Composante")
    ax.set_ylabel("Variance expliquée")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"Variance totale expliquée par {n_components} composantes : **{np.sum(var_ratio):.2%}**")

    st.subheader("🧮 Matrice de confusion — qualité de la projection")
    st.markdown(
        f"On entraîne un classifieur KNN (k=5) **uniquement sur les {n_components} "
        "composantes PCA** (pas sur les 784 pixels d'origine), pour voir quels "
        "chiffres restent confondus après cette réduction de dimension."
    )
    embedding_confusion_matrix(coords, y, class_names)


# ----------------------------------------------------------------------
# LDA
# ----------------------------------------------------------------------
elif section == "LDA":
    st.header("📊 LDA — Linear Discriminant Analysis")

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "LDA a été introduite par **Ronald A. Fisher en 1936** dans son "
            "article fondateur *\"The Use of Multiple Measurements in Taxonomic "
            "Problems\"*, où il cherchait à séparer des espèces de fleurs (le "
            "célèbre dataset Iris) à partir de mesures botaniques. C.R. Rao l'a "
            "généralisée au cas multi-classes en 1948. C'est l'une des toutes "
            "premières méthodes de classification supervisée de l'histoire des "
            "statistiques."
        )
        st.markdown(
            "La LDA est une méthode **supervisée** : contrairement à la PCA, elle "
            "utilise les labels pour trouver les directions qui **séparent le mieux "
            "les classes**, plutôt que celles qui maximisent juste la variance. Elle "
            "peut se voir de deux façons équivalentes : géométrique (critère de "
            "Fisher) ou probabiliste (théorème de Bayes)."
        )

        st.markdown("**① Approche géométrique — critère de Fisher**")
        st.markdown("Maximiser le rapport variance inter-classes / intra-classe :")
        st.latex(r"J(w) = \frac{w^\top S_B\, w}{w^\top S_W\, w}")
        st.markdown("avec la matrice de dispersion **inter-classes** :")
        st.latex(r"S_B = \sum_{k} n_k (\mu_k - \mu)(\mu_k - \mu)^\top")
        st.markdown("et la matrice de dispersion **intra-classe** :")
        st.latex(r"S_W = \sum_{k} \sum_{i \in C_k} (x_i - \mu_k)(x_i - \mu_k)^\top")
        st.markdown("La solution est donnée par le problème aux valeurs propres généralisé :")
        st.latex(r"S_W^{-1} S_B\, w = \lambda w")

        st.markdown("**② Approche probabiliste — théorème de Bayes**")
        st.markdown(
            "LDA peut aussi se dériver entièrement du **théorème de Bayes**, en "
            "supposant que chaque classe suit une loi gaussienne de même "
            "covariance $\\Sigma$. La probabilité a posteriori d'appartenir à la "
            "classe $k$ sachant $x$ :"
        )
        st.latex(
            r"P(y=k \mid x) = \frac{P(x \mid y=k)\, P(y=k)}{P(x)} "
            r"= \frac{P(x \mid y=k)\, \pi_k}{\sum_j P(x \mid y=j)\, \pi_j}"
        )
        st.markdown("avec la densité gaussienne (même $\\Sigma$ pour toutes les classes) :")
        st.latex(
            r"P(x \mid y=k) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} "
            r"\exp\!\left(-\tfrac{1}{2}(x-\mu_k)^\top \Sigma^{-1} (x-\mu_k)\right)"
        )
        st.markdown(
            "En classant $x$ dans la classe qui maximise $P(y=k \\mid x)$, et en "
            "prenant le log du ratio entre deux classes, les termes quadratiques "
            "en $x$ s'annulent (car $\\Sigma$ est commune) : il ne reste qu'une "
            "**fonction linéaire de $x$** — d'où le nom \"Linear\" Discriminant "
            "Analysis :"
        )
        st.latex(
            r"\delta_k(x) = x^\top \Sigma^{-1} \mu_k "
            r"- \tfrac{1}{2}\mu_k^\top \Sigma^{-1} \mu_k + \log \pi_k"
        )
        st.markdown(
            "Les deux approches (Fisher et Bayes) donnent en fait la **même "
            "solution** — c'est l'une des raisons pour lesquelles LDA est un pont "
            "classique entre statistique fréquentiste et bayésienne."
        )
        st.markdown(
            "LDA suppose que toutes les classes partagent la **même matrice de "
            "covariance** — ce qui donne des frontières de décision **linéaires**."
        )

        st.markdown("**③ Le piège en haute dimension — et sa solution (shrinkage)**")
        st.markdown(
            "Sur des images (784 pixels), $\\Sigma$ est une matrice 784×784 : il "
            "faut l'estimer et l'inverser. Si le nombre d'échantillons "
            "d'entraînement est **inférieur** au nombre de pixels (ex. peu "
            "d'images MNIST chargées), cette estimation devient très bruitée, "
            "voire non inversible → l'accuracy s'effondre."
        )
        st.markdown(
            "La solution standard est le **shrinkage** (Ledoit-Wolf) : on "
            "mélange la covariance estimée avec une version simplifiée "
            "(diagonale), pondérée par $\\alpha \\in [0,1]$ :"
        )
        st.latex(r"\hat{\Sigma}_{shrink} = (1-\alpha)\,\hat{\Sigma} + \alpha \, \frac{\mathrm{tr}(\hat{\Sigma})}{d} I")
        st.markdown(
            "Cela stabilise l'estimation et améliore nettement l'accuracy en "
            "haute dimension, au prix d'un petit biais."
        )

    max_components = min(len(class_names) - 1, X_scaled.shape[1])
    n_components = st.slider("Nombre de composantes", 1, max(1, max_components), min(2, max_components))

    st.warning(
        f"⚠️ **Point important** : ce nombre de composantes ({n_components}) "
        "contrôle uniquement le **graphique** (combien d'axes sont calculés "
        "pour la visualisation). Il ne limite **pas** la classification : "
        "`lda.predict()` utilise toujours, en interne, **tous les axes "
        f"discriminants disponibles** (jusqu'à {max_components} ici), même si "
        "tu choisis d'en afficher seulement 2. C'est une particularité de "
        "scikit-learn — l'accuracy affichée plus bas ne changera donc pas "
        "si tu modifies juste ce slider."
    )

    use_shrinkage = st.checkbox(
        "✅ Régularisation shrinkage (recommandé, surtout si peu d'échantillons vs. nb de pixels)",
        value=True,
    )
    if use_shrinkage:
        solver, shrinkage = "eigen", "auto"
    else:
        solver, shrinkage = "svd", None

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# solver="svd" (par défaut) : rapide mais fragile en haute dimension
# solver="eigen", shrinkage="auto" : régularise la covariance (Ledoit-Wolf) -> plus robuste
lda = LinearDiscriminantAnalysis(n_components=n_components, solver=solver, shrinkage=shrinkage)
lda.fit(X_train, y_train)          # utilise LES LABELS pour trouver les axes
y_pred = lda.predict(X_test)

# Projection 2D pour visualisation :
coords_lda = lda.transform(X_scaled)
# coords_lda[:, 0] = axe discriminant 1 (sépare le mieux les classes)
# coords_lda[:, 1] = axe discriminant 2 (2e meilleure direction, orthogonale à la 1re)''',
            language="python",
        )
        st.markdown(
            "Contrairement à `PCA.fit()`, `LDA.fit()` prend `y_train` en argument : "
            "chaque **axe** n'est plus une direction de variance maximale, mais une "
            "direction qui **sépare le mieux les classes** (rapport de Fisher). "
            "C'est pour ça que les points de même couleur se regroupent davantage "
            "sur le graphique LDA que sur le graphique PCA."
        )

    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

    lda = LinearDiscriminantAnalysis(n_components=n_components, solver=solver, shrinkage=shrinkage)
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    show_classification_results(y_test, y_pred, class_names, "LDA")

    if use_shrinkage:
        lda_no_shrink = LinearDiscriminantAnalysis(solver="svd")
        lda_no_shrink.fit(X_train, y_train)
        acc_no_shrink = accuracy_score(y_test, lda_no_shrink.predict(X_test))
        acc_shrink = accuracy_score(y_test, y_pred)
        st.metric(
            "Accuracy avec shrinkage vs sans shrinkage",
            f"{acc_shrink:.2%}",
            delta=f"{acc_shrink - acc_no_shrink:+.2%}",
        )
        st.caption(
            f"Sans shrinkage (solver `svd` par défaut) : {acc_no_shrink:.2%}. "
            "L'écart est surtout visible quand le nombre d'échantillons "
            "d'entraînement est proche ou inférieur au nombre de pixels."
        )

    st.subheader("🔬 Preuve : classification complète vs limitée aux axes affichés")
    st.markdown(
        f"Pour illustrer le point ci-dessus : voici l'accuracy de `lda.predict()` "
        f"(tous les axes) comparée à l'accuracy si on classait en utilisant "
        f"**seulement les {n_components} axes visibles sur le graphique** "
        "(via un classifieur simple du plus proche centroïde, appliqué "
        "directement sur `lda.transform(X)`)."
    )

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''from sklearn.neighbors import NearestCentroid

# 1) La vraie prédiction LDA : utilise TOUS les axes discriminants
acc_full = accuracy_score(y_test, lda.predict(X_test))

# 2) Classification limitée aux n_components axes affichés dans le graphique
X_train_reduced = lda.transform(X_train)     # (n_train, n_components)
X_test_reduced = lda.transform(X_test)       # (n_test, n_components)
clf_reduced = NearestCentroid()
clf_reduced.fit(X_train_reduced, y_train)
acc_reduced = accuracy_score(y_test, clf_reduced.predict(X_test_reduced))''',
            language="python",
        )

    X_train_reduced = lda.transform(X_train)
    X_test_reduced = lda.transform(X_test)
    clf_reduced = NearestCentroid()
    clf_reduced.fit(X_train_reduced, y_train)
    acc_reduced = accuracy_score(y_test, clf_reduced.predict(X_test_reduced))
    acc_full = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy — lda.predict() (tous les axes)", f"{acc_full:.2%}")
    with col2:
        st.metric(
            f"Accuracy — classification sur {n_components} axe(s) seulement",
            f"{acc_reduced:.2%}",
            delta=f"{acc_reduced - acc_full:+.2%}",
        )
    st.caption(
        "Plus tu réduis le nombre de composantes, plus cet écart grandit — "
        "c'est normal : tu perds de l'information discriminante. Le graphique "
        "montre donc une version *simplifiée* de la séparation entre classes, "
        "pas la frontière de décision réellement utilisée par le modèle."
    )

    st.subheader("🖼️ Avant / Après — projection non supervisée (PCA) vs supervisée (LDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔵 Avant — PCA (ne connaît pas les labels)**")
        pca_baseline = PCA(n_components=2, random_state=42)
        coords_pca = pca_baseline.fit_transform(X_scaled)
        plot_2d_projection(coords_pca, y, class_names, "Projection PCA")
    with col2:
        if n_components >= 2:
            st.markdown("**🟢 Après — LDA (utilise les labels)**")
            coords_lda = lda.transform(X_scaled)

            if n_components > 2:
                col_ax1, col_ax2 = st.columns(2)
                with col_ax1:
                    axis_x = st.selectbox("Axe X", list(range(1, n_components + 1)), index=0, key="lda_axis_x")
                with col_ax2:
                    axis_y = st.selectbox("Axe Y", list(range(1, n_components + 1)), index=1, key="lda_axis_y")
            else:
                axis_x, axis_y = 1, 2

            plot_2d_projection(
                coords_lda[:, [axis_x - 1, axis_y - 1]], y, class_names,
                f"Projection LDA (axe discriminant {axis_x} vs {axis_y})"
            )
        else:
            st.info("Une seule composante LDA disponible avec ce nombre de classes.")

    st.caption(
        "ℹ️ **Pourquoi le graphique ne bougeait pas quand tu changais le nombre "
        "de composantes** : LDA calcule ses axes discriminants dans un ordre "
        "fixe (axe 1 = le plus discriminant, axe 2 = le suivant, etc.). "
        "Augmenter le nombre de composantes n'ajoute que des axes "
        "*supplémentaires* (3, 4, ...) — les axes 1 et 2 restent identiques. "
        "Pour voir un graphique différent, choisis d'autres axes ci-dessus "
        "(par ex. axe 3 vs axe 4) une fois que tu as augmenté le nombre de "
        "composantes."
    )


# ----------------------------------------------------------------------
# QDA
# ----------------------------------------------------------------------
elif section == "QDA":
    st.header("📈 QDA — Quadratic Discriminant Analysis")

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "QDA est une extension directe de LDA (Fisher, 1936), formalisée "
            "plus tard dans le cadre général de l'**analyse discriminante "
            "gaussienne** en statistique bayésienne (voir par ex. les travaux de "
            "C.R. Rao dans les années 1940-1950, puis popularisée dans les manuels "
            "de reconnaissance de formes des années 1970-1980, notamment Duda & "
            "Hart, *Pattern Classification*)."
        )
        st.markdown(
            "QDA suit le même principe bayésien que LDA mais **sans supposer une "
            "covariance commune** : chaque classe $k$ a sa propre matrice $\\Sigma_k$."
        )
        st.markdown("Fonction discriminante de la classe $k$ (dérivée du théorème de Bayes, voir section LDA) :")
        st.latex(
            r"\delta_k(x) = -\tfrac{1}{2}\log|\Sigma_k| "
            r"-\tfrac{1}{2}(x-\mu_k)^\top \Sigma_k^{-1} (x-\mu_k) + \log \pi_k"
        )
        st.markdown(
            "où $\\mu_k$ est la moyenne de la classe $k$, $\\pi_k$ sa probabilité "
            "a priori, et $\\Sigma_k$ sa matrice de covariance propre."
        )
        st.markdown(
            "Comme $\\Sigma_k$ diffère selon la classe, le terme quadratique "
            r"$(x-\mu_k)^\top \Sigma_k^{-1} (x-\mu_k)$"
            " ne s'annule pas entre classes → la frontière de décision est "
            "**quadratique** (courbe), contrairement à LDA (droite)."
        )

    st.warning(
        "⚠️ **QDA a besoin d'estimer une matrice de covariance par classe.** "
        "Sur des images brutes (784 pixels pour MNIST complet), il n'y a "
        "généralement pas assez d'échantillons par classe pour que cette "
        "estimation soit possible (la matrice devient singulière). On réduit "
        "donc d'abord la dimension avec une PCA avant d'appliquer QDA."
    )

    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

    # Nombre max de composantes = basé sur le nombre d'échantillons par classe
    # DANS LE TRAIN SET (pas le total), avec une marge de sécurité pour garder
    # une matrice de covariance bien conditionnée (sinon LinAlgError).
    min_class_count = np.bincount(y_train).min()
    max_pca_dim = max(2, min(100, int(min_class_count * 0.6), X_scaled.shape[1]))
    n_components_qda = st.slider(
        "Nombre de composantes PCA avant QDA", 2, max(2, max_pca_dim), min(20, max(2, max_pca_dim))
    )
    reg_param = st.slider("Régularisation (reg_param)", 0.0, 1.0, 0.3, 0.05)
    st.caption(
        f"ℹ️ Classe la moins représentée dans le train set : {min_class_count} échantillons "
        f"→ composantes limitées à {max_pca_dim} pour garder une matrice de covariance stable."
    )

    pca_qda = PCA(n_components=n_components_qda, random_state=42)
    X_train_r = pca_qda.fit_transform(X_train)
    X_test_r = pca_qda.transform(X_test)

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# 1) On réduit d'abord la dimension (indispensable pour QDA sur des images)
pca_qda = PCA(n_components=n_components_qda)
X_train_r = pca_qda.fit_transform(X_train)
X_test_r = pca_qda.transform(X_test)

# 2) QDA estime UNE matrice de covariance par classe (Sigma_k)
qda = QuadraticDiscriminantAnalysis(reg_param=reg_param)
qda.fit(X_train_r, y_train)
y_pred_qda = qda.predict(X_test_r)''',
            language="python",
        )
        st.markdown(
            "La réduction PCA (étape 1) n'est pas là pour visualiser, mais pour "
            "rendre QDA **numériquement possible** : QDA doit inverser une matrice "
            "$\\Sigma_k$ de taille (nb pixels × nb pixels) par classe, ce qui "
            "échoue si on a moins d'échantillons que de pixels dans une classe."
        )

    st.subheader("🔵 Avant — baseline linéaire (LDA, sur les mêmes données réduites)")
    lda_base = LinearDiscriminantAnalysis()
    lda_base.fit(X_train_r, y_train)
    y_pred_lda = lda_base.predict(X_test_r)
    acc_lda = accuracy_score(y_test, y_pred_lda)
    st.metric("Accuracy LDA (frontière linéaire)", f"{acc_lda:.2%}")

    st.subheader("🟢 Après — QDA (frontière quadratique)")
    try:
        qda = QuadraticDiscriminantAnalysis(reg_param=reg_param)
        qda.fit(X_train_r, y_train)
    except np.linalg.LinAlgError as e:
        st.error(
            "QDA n'a pas pu être entraîné même après réduction PCA : "
            f"{e}\n\nEssaie de réduire encore le nombre de composantes ou "
            "d'augmenter reg_param."
        )
        st.stop()
    y_pred_qda = qda.predict(X_test_r)

    show_classification_results(y_test, y_pred_qda, class_names, "QDA")

    acc_qda = accuracy_score(y_test, y_pred_qda)
    delta = acc_qda - acc_lda
    st.metric("Accuracy QDA vs LDA", f"{acc_qda:.2%}", delta=f"{delta:+.2%}")
    st.caption(
        "💡 Sur des données en haute dimension comme MNIST, QDA peut être instable "
        "(pas assez d'échantillons par classe pour estimer $\\Sigma_k$ de façon "
        "fiable) — la régularisation (`reg_param`) aide à stabiliser l'estimation."
    )

    st.subheader("🖼️ Avant / Après — comparaison visuelle (vraies classes vs prédictions QDA)")
    before_after_classification(X_train, X_test, y_test, y_pred_qda, class_names, "QDA")


# ----------------------------------------------------------------------
# t-SNE
# ----------------------------------------------------------------------
elif section == "t-SNE":
    st.header("🌀 t-SNE")

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "t-SNE a été proposé par **Laurens van der Maaten et Geoffrey "
            "Hinton en 2008**, comme amélioration de la méthode SNE (Stochastic "
            "Neighbor Embedding) introduite par **Hinton et Roweis en 2002**. "
            "L'apport principal de t-SNE est l'utilisation d'une distribution "
            "de Student (au lieu d'une gaussienne) en basse dimension, ce qui "
            "règle le \"crowding problem\" (les points s'entassaient trop au "
            "centre avec SNE classique)."
        )
        st.markdown(
            "t-SNE est une méthode **non linéaire** de réduction de dimension, "
            "conçue pour préserver les **voisinages locaux** — des points proches "
            "en haute dimension doivent rester proches en 2D."
        )
        st.markdown("**1. Similarités en haute dimension** (distribution gaussienne autour de $x_i$) :")
        st.latex(
            r"p_{j|i} = \frac{\exp(-\lVert x_i - x_j \rVert^2 / 2\sigma_i^2)}"
            r"{\sum_{k \neq i} \exp(-\lVert x_i - x_k \rVert^2 / 2\sigma_i^2)}"
        )
        st.markdown("**2. Similarités en basse dimension** (distribution t de Student, queues épaisses) :")
        st.latex(
            r"q_{ij} = \frac{(1 + \lVert y_i - y_j \rVert^2)^{-1}}"
            r"{\sum_{k \neq l} (1 + \lVert y_k - y_l \rVert^2)^{-1}}"
        )
        st.markdown("**3. Minimiser la divergence de Kullback-Leibler** entre $P$ et $Q$ :")
        st.latex(r"KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}")
        st.markdown(
            "Le paramètre **perplexity** contrôle le nombre effectif de voisins "
            "considérés autour de chaque point. Contrairement à PCA, t-SNE ne "
            "préserve pas les distances globales — seule la structure locale compte."
        )

    perplexity = st.slider("Perplexity", 5, min(50, X_scaled.shape[0] - 1), 30)
    n_iter = st.slider("Nombre d'itérations", 250, 2000, 1000, step=250)

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Pré-réduction (accélère beaucoup t-SNE sur des données en haute dimension)
X_for_tsne = PCA(n_components=50).fit_transform(X_scaled)

tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, init="pca")
coords_tsne = tsne.fit_transform(X_for_tsne)
# coords_tsne[:, 0], coords_tsne[:, 1] = les 2 axes du graphique
# ATTENTION : ces axes n'ont pas de sens physique (contrairement à PCA), ils
# résultent d'une optimisation itérative — on ne peut pas les interpréter
# comme "direction de variance" ou "direction discriminante".''',
            language="python",
        )
        st.markdown(
            "`fit_transform` ne calcule pas des axes fixes comme PCA : il "
            "**optimise directement les positions 2D** de chaque point (par "
            "descente de gradient) pour que les distances locales en 2D "
            "reflètent au mieux les distances locales en haute dimension."
        )

    X_for_tsne = X_scaled
    if X_scaled.shape[1] > 50:
        pre_reduce = st.checkbox(
            "Pré-réduire avec PCA à 50 dimensions avant t-SNE (recommandé, accélère le calcul)",
            value=True,
        )
        if pre_reduce:
            X_for_tsne = PCA(n_components=50, random_state=42).fit_transform(X_scaled)

    with st.spinner("Calcul de t-SNE en cours... (peut prendre du temps)"):
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42, init="pca")
        coords_tsne = tsne.fit_transform(X_for_tsne)

    st.subheader("🖼️ Avant / Après — PCA (linéaire) vs t-SNE (non linéaire)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔵 Avant — PCA (préserve la variance globale)**")
        coords_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
        plot_2d_projection(coords_pca, y, class_names, "Projection PCA")
    with col2:
        st.markdown("**🟢 Après — t-SNE (préserve les voisinages locaux)**")
        plot_2d_projection(coords_tsne, y, class_names, "Projection t-SNE")
    st.caption(
        "⚠️ Sur t-SNE, les distances entre clusters et les tailles de clusters ne "
        "sont pas directement interprétables — observe surtout si les chiffres "
        "identiques forment des groupes bien séparés des autres."
    )

    st.subheader("🧮 Matrice de confusion — qualité de la projection")
    st.markdown(
        "On entraîne un classifieur KNN (k=5) **uniquement sur les 2 axes t-SNE**, "
        "pour voir quels chiffres restent confondus après cette projection."
    )
    embedding_confusion_matrix(coords_tsne, y, class_names)


# ----------------------------------------------------------------------
# UMAP
# ----------------------------------------------------------------------
elif section == "UMAP":
    st.header("🗺️ UMAP")
    if not UMAP_AVAILABLE:
        st.error("Le package `umap-learn` n'est pas installé. Fais `pip install umap-learn`.")
        st.stop()

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "UMAP a été introduit en **2018** par **Leland McInnes, John Healy "
            "et James Melville** (\"UMAP: Uniform Manifold Approximation and "
            "Projection for Dimension Reduction\"). Il s'appuie sur des "
            "fondations mathématiques issues de la géométrie riemannienne et de "
            "la topologie algébrique (théorie des ensembles flous), et a été "
            "conçu pour être plus rapide que t-SNE tout en préservant mieux la "
            "structure globale des données."
        )
        st.markdown(
            "UMAP repose sur la géométrie riemannienne et la théorie des "
            "ensembles flous. L'idée : les données vivent sur une **variété "
            "(manifold)** de dimension réduite plongée dans l'espace haute "
            "dimension ; UMAP cherche à en apprendre la structure topologique."
        )
        st.markdown("**1. Construction d'un graphe de voisinage flou** — pour chaque point $x_i$, "
                     "probabilité de connexion à son voisin $x_j$ (parmi les $k$ plus proches) :")
        st.latex(
            r"w(x_i, x_j) = \exp\!\left(-\frac{d(x_i, x_j) - \rho_i}{\sigma_i}\right)"
        )
        st.markdown(
            "où $\\rho_i$ est la distance au plus proche voisin (garantit la "
            "connexité locale) et $\\sigma_i$ normalise selon la densité locale."
        )
        st.markdown("**2. Symétrisation** (union floue des deux graphes dirigés) :")
        st.latex(r"w_{ij} = w(x_i,x_j) + w(x_j,x_i) - w(x_i,x_j)\, w(x_j,x_i)")
        st.markdown("**3. Optimisation** — trouver un embedding basse dimension qui minimise "
                     "l'entropie croisée entre le graphe flou haute dimension et son équivalent en 2D :")
        st.latex(
            r"C = \sum_{ij} \Big[ w_{ij} \log \frac{w_{ij}}{v_{ij}} "
            r"+ (1-w_{ij}) \log \frac{1-w_{ij}}{1-v_{ij}} \Big]"
        )
        st.markdown(
            "`n_neighbors` contrôle l'échelle locale vs globale ; `min_dist` "
            "contrôle la compacité des clusters dans l'espace final."
        )

    n_neighbors = st.slider("n_neighbors", 2, min(100, X_scaled.shape[0] - 1), 15)
    min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, 0.05)

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''import umap
from sklearn.decomposition import PCA

X_for_umap = PCA(n_components=50).fit_transform(X_scaled)  # pré-réduction

reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
coords_umap = reducer.fit_transform(X_for_umap)
# coords_umap[:, 0], coords_umap[:, 1] = les 2 axes du graphique
# Comme t-SNE, ces axes viennent d'une optimisation (pas de sens physique direct)''',
            language="python",
        )
        st.markdown(
            "`n_neighbors` fixe combien de voisins sont utilisés pour construire "
            "le graphe flou de départ : petit → structure très locale (fins "
            "détails) ; grand → structure plus globale (vue d'ensemble)."
        )

    X_for_umap = X_scaled
    if X_scaled.shape[1] > 50:
        pre_reduce = st.checkbox(
            "Pré-réduire avec PCA à 50 dimensions avant UMAP (recommandé, accélère le calcul)",
            value=True,
        )
        if pre_reduce:
            X_for_umap = PCA(n_components=50, random_state=42).fit_transform(X_scaled)

    with st.spinner("Calcul de UMAP en cours..."):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
        coords_umap = reducer.fit_transform(X_for_umap)

    st.subheader("🖼️ Avant / Après — PCA (linéaire) vs UMAP (non linéaire)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔵 Avant — PCA**")
        coords_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
        plot_2d_projection(coords_pca, y, class_names, "Projection PCA")
    with col2:
        st.markdown("**🟢 Après — UMAP**")
        plot_2d_projection(coords_umap, y, class_names, "Projection UMAP")

    st.subheader("🧮 Matrice de confusion — qualité de la projection")
    st.markdown(
        "On entraîne un classifieur KNN (k=5) **uniquement sur les 2 axes UMAP**, "
        "pour voir quels chiffres restent confondus après cette projection."
    )
    embedding_confusion_matrix(coords_umap, y, class_names)


# ----------------------------------------------------------------------
# SVM
# ----------------------------------------------------------------------
elif section == "SVM":
    st.header("🧩 SVM — Support Vector Machine")

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "Les fondations théoriques du SVM viennent de **Vladimir Vapnik et "
            "Alexey Chervonenkis** dans les années 1960 (théorie de "
            "l'apprentissage statistique, dimension de Vapnik-Chervonenkis). "
            "L'algorithme à marge maximale a été formalisé en 1992 par **Boser, "
            "Guyon et Vapnik** (introduction du kernel trick), puis étendu à la "
            "**marge souple** par **Cortes et Vapnik en 1995** pour gérer les "
            "données non parfaitement séparables — c'est la version couramment "
            "utilisée aujourd'hui."
        )

        st.markdown(
            "Le SVM cherche l'**hyperplan séparateur qui maximise la marge** "
            "entre les classes."
        )
        st.markdown("**Formulation primale** (marge souple, avec variables de relâchement $\\xi_i$) :")
        st.latex(
            r"\min_{w,b,\xi} \; \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \xi_i "
            r"\quad \text{s.c.} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i,\;\; \xi_i \geq 0"
        )
        st.markdown(
            "$C$ contrôle le compromis entre marge large et erreurs de "
            "classification tolérées."
        )

        st.markdown("**Lagrangien** — on introduit des multiplicateurs de Lagrange "
                     "$\\alpha_i \\geq 0$ pour les contraintes de marge et $\\mu_i \\geq 0$ pour $\\xi_i \\geq 0$ :")
        st.latex(
            r"\mathcal{L}(w,b,\xi,\alpha,\mu) = \tfrac{1}{2}\lVert w \rVert^2 "
            r"+ C\sum_i \xi_i - \sum_i \alpha_i\big[y_i(w^\top x_i + b) - 1 + \xi_i\big] "
            r"- \sum_i \mu_i \xi_i"
        )
        st.markdown(
            "En annulant les dérivées partielles de $\\mathcal{L}$ par rapport à "
            "$w$, $b$ et $\\xi$ (conditions KKT), on obtient :"
        )
        st.latex(r"w = \sum_i \alpha_i y_i x_i \qquad \sum_i \alpha_i y_i = 0 \qquad 0 \le \alpha_i \le C")
        st.markdown(
            "En réinjectant ces relations dans $\\mathcal{L}$, on élimine $w$, $b$ "
            "et $\\xi$ pour obtenir le **problème dual**, qui ne dépend plus que "
            "des $\\alpha_i$ et de produits scalaires entre points :"
        )
        st.latex(
            r"\max_{\alpha} \; \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j\, x_i^\top x_j "
            r"\quad \text{s.c.} \quad 0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0"
        )
        st.markdown(
            "Seuls les points avec $\\alpha_i > 0$ (les **vecteurs de support**) "
            "influencent la frontière — d'où le nom de la méthode. Comme le dual "
            "ne fait intervenir que des produits scalaires $x_i^\\top x_j$, on peut "
            "les remplacer par une fonction noyau (**kernel trick**) pour obtenir "
            "des frontières non linéaires sans jamais calculer $\\varphi(x)$ explicitement :"
        )
        st.latex(r"K(x_i, x_j) = \varphi(x_i)^\top \varphi(x_j)")
        st.markdown("Exemple, noyau RBF (gaussien) :")
        st.latex(r"K(x, x') = \exp\!\left(-\gamma \lVert x - x' \rVert^2\right)")
        st.markdown(
            "Pour un problème multi-classes (10 chiffres), scikit-learn combine "
            "plusieurs SVM binaires (stratégie one-vs-one)."
        )

    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    C = st.slider("C (régularisation)", 0.01, 10.0, 1.0)
    gamma = st.selectbox("Gamma", ["scale", "auto"])

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''from sklearn.svm import SVC

svm = SVC(kernel=kernel, C=C, gamma=gamma)   # kernel="rbf" -> frontière non linéaire
svm.fit(X_train, y_train)                    # résout le problème dual (alpha_i)
y_pred = svm.predict(X_test)

print(svm.n_support_)   # nb de vecteurs de support par classe (les alpha_i > 0)''',
            language="python",
        )
        st.markdown(
            "Le graphique \"avant/après\" ci-dessous n'est **pas** l'espace dans "
            "lequel le SVM travaille (784 dimensions, ou plus avec le kernel RBF) : "
            "c'est une projection PCA à 2 axes utilisée uniquement pour visualiser "
            "où tombent les erreurs — la vraie frontière de décision du SVM vit en "
            "haute dimension et n'est pas représentée directement."
        )

    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

    svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    show_classification_results(y_test, y_pred, class_names, f"SVM ({kernel})")
    st.info(f"Nombre de vecteurs de support : {svm.n_support_.sum()}")

    st.subheader("🖼️ Avant / Après")
    before_after_classification(X_train, X_test, y_test, y_pred, class_names, f"SVM ({kernel})")

    # --------------------------------------------------------------
    # Visualisation de l'hyperplan (frontière de décision) en 2D
    # --------------------------------------------------------------
    st.subheader("🔷 Visualisation de l'hyperplan (frontière de décision)")
    st.markdown(
        "Le vrai SVM travaille en haute dimension (784 pixels ou plus avec le "
        "kernel), impossible à dessiner. Pour **voir** une frontière, on entraîne "
        "ici un second SVM (mêmes hyperparamètres) sur seulement 2 dimensions "
        "(projection PCA), uniquement à des fins pédagogiques."
    )

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''# SVM entraîné uniquement sur 2 dimensions, pour pouvoir tracer la frontière
X_train_2d = PCA(n_components=2).fit_transform(X_train)
svm_2d = SVC(kernel=kernel, C=C, gamma=gamma)
svm_2d.fit(X_train_2d, y_train)

# Grille de points couvrant tout le plan, on prédit la classe de chaque point
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# -> Z donne la couleur de fond (régions de décision) ; les frontières entre
#    couleurs SONT l'hyperplan (ou plutôt sa trace en 2D)''',
            language="python",
        )

    with st.spinner("Calcul de la frontière de décision..."):
        pca_2d_svm = PCA(n_components=2, random_state=42)
        X_train_2d = pca_2d_svm.fit_transform(X_train)

        svm_2d = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        svm_2d.fit(X_train_2d, y_train)

        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="tab10", levels=np.arange(-0.5, len(class_names) + 0.5, 1))
    scatter = ax.scatter(
        X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap="tab10",
        s=18, edgecolor="k", linewidth=0.3, alpha=0.9
    )
    sv = svm_2d.support_vectors_
    ax.scatter(
        sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolors="black",
        linewidths=1.5, label=f"Vecteurs de support ({len(sv)})"
    )
    handles, _ = scatter.legend_elements()
    ax.legend(handles + [ax.collections[-1]], list(class_names) + ["Vecteurs de support"],
              bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xlabel("Composante principale 1 (PCA)")
    ax.set_ylabel("Composante principale 2 (PCA)")
    ax.set_title(f"Frontière de décision SVM ({kernel}) — modèle 2D pour visualisation")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Les zones colorées sont les régions de décision ; leurs frontières "
        "matérialisent l'hyperplan (ou sa version courbée avec un kernel non "
        "linéaire). Les points entourés en noir sont les vecteurs de support : "
        "ce sont eux, et eux seuls, qui déterminent la position de la frontière."
    )

    # --------------------------------------------------------------
    # Augmentation de dimension (kernel trick) — démo illustrative
    # --------------------------------------------------------------
    st.subheader("🧬 Augmentation de dimension — pourquoi le kernel trick marche")
    st.markdown(
        "Cas d'école (données synthétiques, pas MNIST) : deux classes en "
        "anneaux concentriques. **Impossible à séparer par une droite en 2D**, "
        "mais en ajoutant une 3ᵉ dimension $z = x^2 + y^2$, les deux classes "
        "se séparent parfaitement par un simple **plan horizontal**. C'est "
        "exactement ce que fait implicitement le kernel RBF, mais dans un "
        "espace à un nombre de dimensions infini, sans jamais calculer $\\varphi(x)$."
    )

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''# Données non séparables linéairement en 2D (anneaux concentriques)
theta = np.random.uniform(0, 2*np.pi, n)
X_inner = r_inner * [cos(theta), sin(theta)]     # classe 0 : petit anneau
X_outer = r_outer * [cos(theta), sin(theta)]     # classe 1 : grand anneau

# Augmentation de dimension : on ajoute une coordonnée z
z = x**2 + y**2   # distance au centre, au carré

# En 3D (x, y, z), un simple plan z = seuil sépare parfaitement les 2 classes
# -> c'est l'idée du kernel trick : projeter dans un espace où ça devient linéaire''',
            language="python",
        )

    rng_demo = np.random.RandomState(42)
    n_demo = 150
    theta_demo = rng_demo.uniform(0, 2 * np.pi, n_demo)
    r_inner = rng_demo.normal(1.0, 0.1, n_demo)
    r_outer = rng_demo.normal(3.0, 0.15, n_demo)
    X_inner_demo = np.c_[r_inner * np.cos(theta_demo), r_inner * np.sin(theta_demo)]
    X_outer_demo = np.c_[r_outer * np.cos(theta_demo), r_outer * np.sin(theta_demo)]
    X_demo = np.vstack([X_inner_demo, X_outer_demo])
    y_demo = np.array([0] * n_demo + [1] * n_demo)
    z_demo = X_demo[:, 0] ** 2 + X_demo[:, 1] ** 2

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔵 Avant — en 2D (non séparable par une droite)**")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(X_demo[:, 0], X_demo[:, 1], c=y_demo, cmap="coolwarm", s=25, edgecolor="k", linewidth=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("2 dimensions : (x, y)")
        ax.set_aspect("equal")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.markdown("**🟢 Après — en 3D avec $z = x^2 + y^2$ (séparable par un plan)**")
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(X_demo[:, 0], X_demo[:, 1], z_demo, c=y_demo, cmap="coolwarm", s=25, edgecolor="k", linewidth=0.3)
        xx_plane, yy_plane = np.meshgrid(np.linspace(-4, 4, 10), np.linspace(-4, 4, 10))
        zz_plane = np.full_like(xx_plane, 4.5)
        ax.plot_surface(xx_plane, yy_plane, zz_plane, alpha=0.25, color="gray")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z = x² + y²")
        ax.set_title("3 dimensions : (x, y, x²+y²)")
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------
# One-Class SVM
# ----------------------------------------------------------------------
elif section == "One-Class SVM":
    st.header("🎯 One-Class SVM — Détection d'anomalies")

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "One-Class SVM a été proposé par **Bernhard Schölkopf, John "
            "Platt, John Shawe-Taylor, Alex Smola et Robert Williamson en 2001** "
            "(\"Estimating the Support of a High-Dimensional Distribution\"), "
            "comme adaptation du SVM classique au problème non supervisé "
            "d'estimation de support d'une distribution / détection d'anomalies."
        )
        st.markdown(
            "One-Class SVM apprend la frontière d'une seule classe \"normale\" "
            "et détecte ensuite les points qui s'en écartent (anomalies / outliers)."
        )
        st.markdown(
            "Il cherche un hyperplan séparant les données de l'origine avec "
            "**marge maximale** dans l'espace du noyau :"
        )
        st.latex(
            r"\min_{w,\rho,\xi} \; \tfrac{1}{2}\lVert w \rVert^2 "
            r"+ \tfrac{1}{\nu n}\sum_i \xi_i - \rho "
            r"\quad \text{s.c.} \quad w^\top \varphi(x_i) \geq \rho - \xi_i,\;\; \xi_i \geq 0"
        )
        st.markdown("La fonction de décision est :")
        st.latex(r"f(x) = \operatorname{sign}\big(w^\top \varphi(x) - \rho\big)")
        st.markdown(
            "$\\nu \\in (0,1]$ borne à la fois la fraction d'outliers tolérée "
            "dans les données d'entraînement et la fraction minimale de vecteurs "
            "de support."
        )

    normal_class = st.selectbox("Chiffre considéré comme 'normal'", options=list(class_names))
    normal_idx = list(class_names).index(normal_class)

    nu = st.slider("nu (proportion d'outliers attendue)", 0.01, 0.5, 0.1, 0.01)
    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    gamma = st.selectbox("Gamma", ["scale", "auto"])

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''from sklearn.svm import OneClassSVM

# Entraîné UNIQUEMENT sur les images du chiffre "normal"
X_normal = X_scaled[y == normal_idx]
ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
ocsvm.fit(X_normal)

# Appliqué ensuite à TOUTES les images (y compris les autres chiffres)
preds = ocsvm.predict(X_scaled)   # +1 = normal, -1 = anomalie''',
            language="python",
        )
        st.markdown(
            "Le modèle n'a jamais vu les autres chiffres pendant l'entraînement : "
            "il apprend seulement \"à quoi ressemble\" le chiffre normal, puis "
            "signale comme anomalie tout ce qui s'en écarte trop — c'est pour ça "
            "qu'on peut l'utiliser pour détecter des formes jamais vues."
        )

    X_normal = X_scaled[y == normal_idx]
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    ocsvm.fit(X_normal)
    preds = ocsvm.predict(X_scaled)  # 1 = normal, -1 = anomalie

    pca_viz = PCA(n_components=2, random_state=42)
    coords = pca_viz.fit_transform(X_scaled)

    st.subheader("🖼️ Avant / Après")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔵 Avant — tous les points (statut inconnu)**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(coords[:, 0], coords[:, 1], c="steelblue", s=18, alpha=0.6)
        ax.set_title("Avant classification (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.markdown(f"**🟢 Après — normal (chiffre {normal_class}) vs anomalie**")
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = np.where(preds == 1, "steelblue", "crimson")
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, alpha=0.8)
        ax.set_title("Détection d'anomalies (One-Class SVM)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(
            handles=[
                mpatches.Patch(color="steelblue", label="Normal"),
                mpatches.Patch(color="crimson", label="Anomalie"),
            ]
        )
        st.pyplot(fig)
        plt.close(fig)

    n_anomalies = int(np.sum(preds == -1))
    st.metric("Anomalies détectées", f"{n_anomalies} / {len(preds)}")

    st.subheader("🧮 Matrice de confusion — Normal vs Anomalie")
    st.markdown(
        f"Le modèle n'a **jamais vu** les labels des autres chiffres pendant "
        f"l'entraînement (il n'a vu que des images du chiffre '{normal_class}'). "
        "On peut malgré tout évaluer *a posteriori* sa performance en comparant "
        f"ses prédictions à la vraie règle \"est-ce le chiffre {normal_class} ou non ?\"."
    )
    y_true_binary = np.where(y == normal_idx, "Normal", "Anomalie")
    y_pred_binary = np.where(preds == 1, "Normal", "Anomalie")

    col1, col2 = st.columns([1, 2])
    with col1:
        acc_binary = accuracy_score(y_true_binary, y_pred_binary)
        st.metric("Accuracy (Normal vs Anomalie)", f"{acc_binary:.2%}")
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary, labels=["Normal", "Anomalie"])
        ConfusionMatrixDisplay(cm_binary, display_labels=["Normal", "Anomalie"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------
# Decision Tree
# ----------------------------------------------------------------------
elif section == "Decision Tree":
    st.header("🌳 Decision Tree")

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "Les arbres de décision modernes viennent de deux lignées : "
            "**ID3**, développé par **Ross Quinlan en 1986** (utilisant "
            "l'entropie / le gain d'information), et **CART** (Classification "
            "And Regression Trees), publié par **Breiman, Friedman, Olshen et "
            "Stone en 1984** (utilisant l'impureté de Gini). scikit-learn "
            "implémente une version optimisée de CART."
        )
        st.markdown(
            "Un arbre de décision découpe récursivement l'espace des données en "
            "posant des questions du type \"pixel $i$ > seuil ?\", en choisissant "
            "à chaque étape la coupure qui **réduit le plus l'impureté**."
        )
        st.markdown("**Impureté de Gini** (probabilité de mal classer un point tiré au hasard) :")
        st.latex(r"Gini(t) = 1 - \sum_{k} p_k^2")
        st.markdown("**Entropie** (mesure théorie de l'information) :")
        st.latex(r"H(t) = -\sum_{k} p_k \log_2 p_k")
        st.markdown("**Gain d'information** d'une coupure séparant un nœud $t$ en $t_L$ et $t_R$ :")
        st.latex(r"\Delta I = I(t) - \frac{n_L}{n} I(t_L) - \frac{n_R}{n} I(t_R)")
        st.markdown(
            "où $p_k$ est la proportion de la classe $k$ dans le nœud, et $I$ "
            "peut être le Gini ou l'entropie. L'algorithme choisit à chaque nœud "
            "la variable et le seuil qui maximisent $\\Delta I$."
        )

    max_depth = st.slider("Profondeur max", 1, 15, 4)
    criterion = st.selectbox("Critère", ["gini", "entropy", "log_loss"])

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# Importance de chaque pixel dans les décisions de l'arbre :
importances = tree.feature_importances_   # un score par pixel (784 valeurs)''',
            language="python",
        )
        st.markdown(
            "`feature_importances_` mesure, pour chaque pixel, à quel point il a "
            "contribué à réduire l'impureté (Gini/entropie) sur l'ensemble de "
            "l'arbre — les pixels du centre de l'image sont souvent les plus "
            "importants (les coins sont presque toujours noirs, donc inutiles)."
        )

    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

    tree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    show_classification_results(y_test, y_pred, class_names, "Decision Tree")

    st.subheader("🖼️ Avant / Après")
    before_after_classification(X_train, X_test, y_test, y_pred, class_names, "Decision Tree")

    st.subheader("Visualisation de l'arbre (profondeur limitée à 3 pour la lisibilité)")
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        tree, feature_names=[f"px{c}" for c in pixel_cols], class_names=[str(c) for c in class_names],
        filled=True, rounded=True, fontsize=8, ax=ax, max_depth=3
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Importance des variables (pixels les plus discriminants)")
    importances = pd.Series(tree.feature_importances_, index=pixel_cols).sort_values(ascending=False).head(20)
    st.bar_chart(importances)


# ----------------------------------------------------------------------
# KNN
# ----------------------------------------------------------------------
elif section == "KNN":
    st.header("👥 KNN — K-Nearest Neighbors")

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "KNN a été introduit par **Evelyn Fix et Joseph Hodges en 1951** "
            "dans un rapport technique de l'US Air Force sur la discrimination "
            "non paramétrique, puis formalisé et popularisé par **Thomas Cover "
            "et Peter Hart en 1967**, qui ont démontré ses propriétés théoriques "
            "(notamment que son taux d'erreur asymptotique est borné par deux "
            "fois le taux d'erreur bayésien optimal)."
        )
        st.markdown(
            "KNN est un algorithme **paresseux** (\"lazy learner\") : il ne "
            "construit aucun modèle à l'entraînement, il stocke simplement les "
            "données. Pour classer un nouveau point $x$, il regarde ses $k$ plus "
            "proches voisins dans le jeu d'entraînement et prend un **vote majoritaire**."
        )
        st.markdown("**Distance euclidienne** (par défaut) entre deux images $x$ et $x'$ (vecteurs de 784 pixels) :")
        st.latex(r"d(x, x') = \sqrt{\sum_{i=1}^{d} (x_i - x_i')^2}")
        st.markdown("**Règle de décision** — classe majoritaire parmi les $k$ plus proches voisins $N_k(x)$ :")
        st.latex(r"\hat{y} = \arg\max_{c} \sum_{i \in N_k(x)} \mathbb{1}(y_i = c)")
        st.markdown(
            "Un $k$ petit donne une frontière de décision très sensible au bruit "
            "(overfitting) ; un $k$ grand lisse la frontière mais peut mélanger "
            "des classes différentes (underfitting)."
        )

    k = st.slider("Nombre de voisins (k)", 1, 30, 5)
    weights = st.selectbox("Pondération", ["uniform", "distance"])
    metric = st.selectbox("Distance", ["minkowski", "euclidean", "manhattan"])

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
knn.fit(X_train, y_train)          # ne fait qu'enregistrer les données (lazy)
y_pred = knn.predict(X_test)

# Retrouver les k voisins d'UNE image précise (pour la visu "avant/après") :
distances, positions = knn.kneighbors(X_test[query_idx].reshape(1, -1), n_neighbors=k)''',
            language="python",
        )
        st.markdown(
            "`kneighbors()` renvoie les distances et les indices des `k` images "
            "d'entraînement les plus proches de l'image interrogée — c'est "
            "exactement le calcul qui alimente le graphique \"les k plus proches "
            "voisins\" ci-dessous."
        )

    # Split par indices pour pouvoir retrouver les images brutes correspondantes
    indices = np.arange(len(X_scaled))
    idx_train, idx_test = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    X_train, X_test = X_scaled[idx_train], X_scaled[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    X_train_pixels, X_test_pixels = X_raw.values[idx_train], X_raw.values[idx_test]

    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    show_classification_results(y_test, y_pred, class_names, "KNN")

    query_idx = st.slider("Choisir une image de test", 0, len(X_test) - 1, 0)
    distances, neighbor_positions = knn.kneighbors(X_test[query_idx].reshape(1, -1), n_neighbors=k)
    neighbor_positions = neighbor_positions[0]
    distances = distances[0]

    # ----------------------------------------------------------------
    # Graphique "mécanisme" : cercle de voisinage + vote, comme une
    # coupe zoomée sur le point interrogé (vue simplifiée en 2D PCA,
    # pour que le cercle représente géométriquement les vrais voisins).
    # ----------------------------------------------------------------
    st.subheader("🔍 Mécanisme du vote KNN — vue zoomée")
    st.markdown(
        "Le vrai KNN vote dans l'espace complet (784 pixels). Pour que le "
        "**cercle de voisinage** soit géométriquement exact sur un graphique 2D, "
        "on recalcule ici KNN directement dans l'espace **PCA à 2 dimensions** "
        "(version simplifiée, à but pédagogique)."
    )

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''# KNN recalculé en 2D (PCA) pour que le cercle de voisinage soit géométriquement exact
X_train_2d = PCA(n_components=2).fit_transform(X_train)
X_test_2d = PCA(n_components=2).fit(X_train).transform(X_test)

knn_2d = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
knn_2d.fit(X_train_2d, y_train)

distances, positions = knn_2d.kneighbors(X_test_2d[query_idx].reshape(1, -1), n_neighbors=k)
prediction = knn_2d.predict(X_test_2d[query_idx].reshape(1, -1))
# distances.max() = rayon du cercle de voisinage tracé sur le graphique''',
            language="python",
        )

    pca_2d_knn = PCA(n_components=2, random_state=42).fit(X_train)
    X_train_2d = pca_2d_knn.transform(X_train)
    X_test_2d = pca_2d_knn.transform(X_test)

    knn_2d = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
    knn_2d.fit(X_train_2d, y_train)

    query_point = X_test_2d[query_idx].reshape(1, -1)
    dist_2d, pos_2d = knn_2d.kneighbors(query_point, n_neighbors=k)
    dist_2d, pos_2d = dist_2d[0], pos_2d[0]
    neighbor_labels_2d = y_train[pos_2d]
    pred_2d = knn_2d.predict(query_point)[0]

    vals, counts = np.unique(neighbor_labels_2d, return_counts=True)
    order = np.argsort(-counts)
    vote_str = ", ".join(f"{class_names[vals[i]]}:{counts[i]}x" for i in order)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c="lightgray", s=15, alpha=0.4, zorder=1)

    radius = dist_2d.max()
    circle = plt.Circle(
        (query_point[0, 0], query_point[0, 1]), radius,
        fill=False, edgecolor="red", linestyle=":", linewidth=2, zorder=2
    )
    ax.add_patch(circle)

    cmap = plt.colormaps.get_cmap("tab10")
    for pos, dist in zip(pos_2d, dist_2d):
        px, py = X_train_2d[pos]
        ax.plot([query_point[0, 0], px], [query_point[0, 1], py], "k--", linewidth=0.8, zorder=2)
        label = y_train[pos]
        ax.scatter(px, py, s=380, color=cmap(label / 10), edgecolor="black", linewidth=1.5, zorder=3)
        ax.text(px, py, str(class_names[label]), color="white", fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=4)

    ax.scatter(
        query_point[0, 0], query_point[0, 1], marker="*", s=600, color="red",
        edgecolor="black", linewidth=1, zorder=5,
        label=f"Point à classer (vrai = {class_names[y_test[query_idx]]})"
    )

    margin = radius * 1.3 if radius > 0 else 1
    ax.set_xlim(query_point[0, 0] - margin, query_point[0, 0] + margin)
    ax.set_ylim(query_point[0, 1] - margin, query_point[0, 1] + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("Composante principale 1 (PCA)")
    ax.set_ylabel("Composante principale 2 (PCA)")
    ax.set_title(
        f"Mécanisme KNN (k={k}) — zoom sur le point\n"
        f"Vote des {k} voisins : {vote_str}  →  Prédiction = {class_names[pred_2d]}  "
        f"(vrai label = {class_names[y_test[query_idx]]})"
    )
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "💡 Cette version 2D est simplifiée pour la pédagogie : la prédiction du "
        "graphique (`{}`.) peut différer de celle du vrai modèle KNN "
        "(entraîné sur 784 pixels, indiquée dans les métriques plus haut) — "
        "c'est normal, l'espace PCA 2D ne conserve pas toutes les distances "
        "d'origine.".format(class_names[pred_2d])
    )

    st.subheader("🖼️ Avant / Après — projection PCA (vraies classes vs prédictions)")
    before_after_classification(X_train, X_test, y_test, y_pred, class_names, f"KNN (k={k})")

    st.subheader("🔎 Avant / Après — image requête et ses k plus proches voisins")
    st.markdown(
        "Ici, \"avant\" = l'image que l'on veut classer (seule, sans label connu), "
        "\"après\" = les k images les plus proches trouvées dans les données "
        "d'entraînement, qui déterminent la prédiction par vote majoritaire "
        "(cette fois calculées dans l'espace réel à 784 pixels)."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**🔵 Avant — image à classer**")
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(X_test_pixels[query_idx].reshape(image_shape), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Vraie classe : {class_names[y_test[query_idx]]}")
        st.pyplot(fig)
        plt.close(fig)
        pred_label = knn.predict(X_test[query_idx].reshape(1, -1))[0]
        st.metric("Classe prédite (vote des k voisins)", str(class_names[pred_label]))

    with col2:
        st.markdown(f"**🟢 Après — les {k} plus proches voisins (utilisés pour le vote)**")
        n_cols = min(k, 10)
        n_rows = int(np.ceil(k / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.8 * n_rows))
        axes = np.array(axes).reshape(-1)
        for i in range(n_rows * n_cols):
            ax = axes[i]
            if i < k:
                pos = neighbor_positions[i]
                ax.imshow(X_train_pixels[pos].reshape(image_shape), cmap="gray")
                is_same = y_train[pos] == y_test[query_idx]
                color = "green" if is_same else "red"
                ax.set_title(f"{class_names[y_train[pos]]}\nd={distances[i]:.1f}", fontsize=8, color=color)
            ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Accuracy en fonction de k")
    ks = range(1, 31)
    accs = []
    for kk in ks:
        m = KNeighborsClassifier(n_neighbors=kk, weights=weights, metric=metric)
        m.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, m.predict(X_test)))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(list(ks), accs, marker="o")
    ax.axvline(k, color="red", linestyle="--", label=f"k choisi = {k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------
# Compressive Sensing
# ----------------------------------------------------------------------
elif section == "Compressive Sensing":
    st.header("📡 Compressive Sensing — Échantillonnage compressé")

    with st.expander("📐 Théorie et mathématiques", expanded=True):
        st.markdown("**📜 Origine**")
        st.markdown(
            "Développé par **Emmanuel Candès, Justin Romberg et Terence Tao**, "
            "et indépendamment par **David Donoho**, entre 2004 et 2006. Tao a "
            "reçu la médaille Fields en 2006, en partie pour ces travaux. C'est "
            "l'une des avancées les plus marquantes du traitement du signal "
            "moderne."
        )
        st.markdown(
            "**Idée centrale** : le théorème de Shannon-Nyquist dit qu'il faut "
            "échantillonner à au moins 2× la fréquence maximale d'un signal "
            "pour le reconstruire parfaitement. Le compressive sensing casse "
            "cette règle : si le signal est **parcimonieux** (peu de "
            "coefficients non nuls dans une certaine base), on peut le "
            "reconstruire avec **beaucoup moins de mesures**."
        )
        st.markdown("**Modèle de mesure** — au lieu d'observer $x \\in \\mathbb{R}^n$ directement, "
                     "on prend $m \\ll n$ mesures linéaires aléatoires :")
        st.latex(r"y = \Phi x, \qquad \Phi \in \mathbb{R}^{m \times n},\ y \in \mathbb{R}^m")
        st.markdown(
            "Sur une image MNIST, les pixels bruts ne sont pas toujours assez "
            "parcimonieux — mais l'image devient très parcimonieuse dans la "
            "**base DCT** (transformée en cosinus discrète, la même que celle "
            "utilisée par le format JPEG) : $x = \\Psi s$, où $s$ n'a que "
            "quelques coefficients significatifs."
        )
        st.markdown("**Reconstruction idéale** — trouver le signal le plus parcimonieux "
                     "compatible avec les mesures (problème $\\ell_0$, NP-difficile) :")
        st.latex(r"\min_{s} \|s\|_0 \quad \text{s.c.} \quad y = \Phi \Psi s")
        st.markdown(
            "**Résultat clé du compressive sensing** : sous la propriété "
            "d'isométrie restreinte (RIP) de $\\Phi$ (vérifiée avec haute "
            "probabilité par des matrices aléatoires gaussiennes), la "
            "relaxation convexe en norme $\\ell_1$ retrouve la **même "
            "solution** :"
        )
        st.latex(r"\min_{s} \|s\|_1 \quad \text{s.c.} \quad y = \Phi \Psi s")
        st.markdown(
            "avec seulement $m = O(k \\log(n/k))$ mesures ($k$ = nombre de "
            "coefficients non nuls). En pratique (avec du bruit), on résout "
            "la version régularisée (LASSO) :"
        )
        st.latex(r"\min_{s} \tfrac{1}{2}\|y - \Phi\Psi s\|_2^2 + \alpha \|s\|_1")
        st.markdown(
            "**Pourquoi $\\ell_1$ et pas $\\ell_2$ ?** La norme $\\ell_2$ "
            "(moindres carrés / pseudo-inverse) étale l'erreur sur tous les "
            "coefficients et ne produit jamais de vrais zéros. La norme "
            "$\\ell_1$ favorise géométriquement les solutions parcimonieuses "
            "(beaucoup de coefficients exactement nuls) — c'est ce qui permet "
            "de retrouver le signal d'origine."
        )

    n_pixels = image_shape[0] * image_shape[1]
    idx_cs = st.slider("Choisir une image", 0, len(X_raw) - 1, 0, key="cs_idx")
    m = st.slider(
        "Nombre de mesures m (compression)", 10, n_pixels,
        min(max(20, n_pixels // 5), n_pixels), step=5
    )
    alpha_cs = st.slider("Régularisation L1 / Lasso (alpha)", 0.0001, 0.02, 0.001, format="%.4f")
    st.write(f"Taux de compression : **{m} mesures / {n_pixels} pixels = {m / n_pixels:.1%}**")

    with st.expander("💻 Le code correspondant", expanded=False):
        st.code(
            '''from scipy.fftpack import dct, idct
from sklearn.linear_model import Lasso

# Psi = base IDCT (transforme des coefficients DCT parcimonieux en image)
Psi = build_idct_basis(image_shape)          # matrice (n_pixels, n_pixels)

# Matrice de mesure ALEATOIRE (indépendante des données !)
Phi = np.random.randn(m, n_pixels) / np.sqrt(m)
y = Phi @ x_true                              # y : m mesures compressées

# Reconstruction naive (norme L2 minimale, pseudo-inverse) -> mauvaise
x_l2 = np.linalg.pinv(Phi) @ y

# Reconstruction L1 (Lasso) dans la base DCT, PUIS on revient aux pixels
A = Phi @ Psi                                 # opérateur combiné (m, n_pixels)
lasso = Lasso(alpha=alpha)
lasso.fit(A, y)                               # trouve les coefficients DCT s
x_l1 = Psi @ lasso.coef_                      # image reconstruite = Psi @ s''',
            language="python",
        )

    with st.spinner("Reconstruction en cours..."):
        Psi = build_idct_basis(image_shape)

        img_cs = X_raw.values[idx_cs].reshape(image_shape).astype(float)
        img_cs = img_cs / (img_cs.max() + 1e-8)
        x_true = img_cs.flatten()

        rng_cs = np.random.RandomState(0)
        Phi = rng_cs.randn(m, n_pixels) / np.sqrt(m)
        y_meas = Phi @ x_true

        x_l2 = np.linalg.pinv(Phi) @ y_meas

        A = Phi @ Psi
        lasso_cs = Lasso(alpha=alpha_cs, max_iter=5000)
        lasso_cs.fit(A, y_meas)
        x_l1 = Psi @ lasso_cs.coef_

    mse_l2 = np.mean((x_l2 - x_true) ** 2)
    mse_l1 = np.mean((x_l1 - x_true) ** 2)

    st.subheader("🖼️ Avant / Après — reconstruction depuis mesures compressées")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔵 Avant — image originale**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(x_true.reshape(image_shape), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{n_pixels} pixels")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.markdown("**❌ Reconstruction L2 (naïve)**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(x_l2.reshape(image_shape), cmap="gray")
        ax.axis("off")
        ax.set_title(f"MSE = {mse_l2:.4f}")
        st.pyplot(fig)
        plt.close(fig)
    with col3:
        st.markdown("**🟢 Après — Reconstruction L1 (Lasso, base DCT)**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(x_l1.reshape(image_shape), cmap="gray")
        ax.axis("off")
        ax.set_title(f"MSE = {mse_l1:.4f}")
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "La reconstruction **L2** (pseudo-inverse) étale l'erreur sur toute "
        "l'image et reste floue/bruitée dès que $m < n$. La reconstruction "
        "**L1** exploite la parcimonie du signal dans la base DCT et retrouve "
        "une image bien plus fidèle, même avec beaucoup moins de mesures que "
        "de pixels. Si ce n'est pas flagrant, essaie d'augmenter $m$ ou "
        "d'ajuster `alpha` (plus petit = reconstruction plus fidèle mais plus "
        "sensible au bruit)."
    )

    st.subheader("📈 Qualité de reconstruction en fonction du nombre de mesures")
    with st.spinner("Calcul de la courbe (plusieurs valeurs de m)..."):
        m_values = np.unique(np.linspace(15, n_pixels, 8, dtype=int))
        mse_l2_list, mse_l1_list = [], []
        for mv in m_values:
            rng_mv = np.random.RandomState(42)
            Phi_v = rng_mv.randn(mv, n_pixels) / np.sqrt(mv)
            y_v = Phi_v @ x_true

            x_l2_v = np.linalg.pinv(Phi_v) @ y_v
            mse_l2_list.append(np.mean((x_l2_v - x_true) ** 2))

            A_v = Phi_v @ Psi
            lasso_v = Lasso(alpha=alpha_cs, max_iter=3000)
            lasso_v.fit(A_v, y_v)
            x_l1_v = Psi @ lasso_v.coef_
            mse_l1_list.append(np.mean((x_l1_v - x_true) ** 2))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(m_values, mse_l2_list, marker="o", label="L2 (naïve)", color="crimson")
    ax.plot(m_values, mse_l1_list, marker="o", label="L1 (Lasso, DCT)", color="steelblue")
    ax.axvline(m, color="gray", linestyle="--", label=f"m choisi = {m}")
    ax.set_xlabel("Nombre de mesures m")
    ax.set_ylabel("Erreur de reconstruction (MSE)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "💡 L'avantage de L1 est surtout net dans la zone de **forte "
        "compression** (m petit devant n). Avec un `alpha` fixe, L2 peut "
        "parfois rattraper L1 quand m devient grand (proche de n) — en "
        "pratique, `alpha` devrait être réduit à mesure que m augmente "
        "(moins besoin de régularisation avec plus de mesures)."
    )


st.sidebar.divider()
st.sidebar.caption("App créée avec Streamlit + scikit-learn — données MNIST")