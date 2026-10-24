# TP1 : Initiation à Python
## POO via Python - 2025/2026
**Préparé par :** Pr. Miloudi Chaymae

---

## 📁 Structure des fichiers

- `exercice1.py` - Manipulation de 3 valeurs entières
- `exercice2.py` - Résolution d'équation du second degré
- `exercice3.py` - Agence de location de voitures
- `exercice4.py` - Saisie sécurisée (nombre positif)
- `exercice5.py` - Table de multiplication
- `exercice6.py` - Calcul des bénéfices d'entreprise
- `tp1_menu.py` - Menu principal pour exécuter tous les exercices
- `README.md` - Ce fichier

---

## 🚀 Comment utiliser ces fichiers dans VS Code

### Option 1 : Exécuter un exercice individuel
1. Ouvrez VS Code
2. Ouvrez le fichier de l'exercice souhaité (ex: `exercice1.py`)
3. Appuyez sur **F5** ou cliquez sur le bouton ▶️ "Run Python File"
4. Suivez les instructions dans le terminal

### Option 2 : Utiliser le menu principal
1. Ouvrez le fichier `tp1_menu.py`
2. Exécutez-le (F5 ou ▶️)
3. Choisissez l'exercice à exécuter via le menu interactif

---

## 📝 Description des exercices

### Exercice 1 : Manipulation de 3 valeurs
- Lire 3 valeurs entières au clavier
- Calculer et afficher : somme, moyenne, produit
- Trier et afficher les valeurs en ordre croissant

### Exercice 2 : Résolution d'équation
- Résoudre l'équation : ax² + bx + c = 0
- Vérifier l'existence de racines réelles
- Calculer les valeurs des racines si elles existent
- Utilise la fonction `sqrt()` de la bibliothèque `math`

### Exercice 3 : Location de voitures
Tarifs disponibles :
- **Essence** : 150 MAD/jour + 1,20 MAD/km
- **Diesel** : 220 MAD/jour + 0,70 MAD/km

Le programme calcule le meilleur choix selon la distance et la durée.

### Exercice 4 : Saisie sécurisée
- Saisie uniquement de nombres positifs
- Utilise une boucle `while`
- Continue à demander jusqu'à obtenir un nombre valide

### Exercice 5 : Table de multiplication
- Affiche la table de multiplication d'un entier
- Utilise une boucle `for`
- Affichage de 1 à 10

### Exercice 6 : Bénéfices d'entreprise
Détermine si l'entreprise réalise :
- Un **bénéfice** (revenu > coût)
- Une **perte** (revenu < coût)
- Le **seuil de rentabilité** (revenu = coût)

---

## 💡 Conseils pour VS Code

### Installation de Python
Si Python n'est pas installé :
1. Téléchargez Python sur [python.org](https://www.python.org/downloads/)
2. Cochez "Add Python to PATH" lors de l'installation

### Extension VS Code recommandée
- **Python** (de Microsoft) : recherchez "Python" dans les extensions (Ctrl+Shift+X)

### Raccourcis utiles
- **F5** : Exécuter le fichier Python
- **Ctrl+`** : Ouvrir/fermer le terminal
- **Ctrl+S** : Sauvegarder le fichier

---

## 🎯 Comment tester vos programmes

### Exemples de données de test

**Exercice 1 :**
```
Valeurs : 10, 5, 8
Résultat attendu : Somme=23, Moyenne=7.67, Produit=400, Trié=[5, 8, 10]
```

**Exercice 2 :**
```
a=1, b=-5, c=6
Résultat attendu : Deux racines (x₁=3, x₂=2)
```

**Exercice 3 :**
```
Distance=100 km, Durée=2 jours
Essence: 150×2 + 1.20×100 = 420 MAD
Diesel: 220×2 + 0.70×100 = 510 MAD
Meilleur choix : Essence
```

**Exercice 5 :**
```
Entier : 7
Table de 1×7 jusqu'à 10×7
```

**Exercice 6 :**
```
Coût unitaire=50, Prix vente=70, Quantité=100
Bénéfice = (70-50)×100 = 2000 MAD
```

---

## ❓ Aide et support

Si vous rencontrez des problèmes :
1. Vérifiez que Python est bien installé : `python --version`
2. Vérifiez que vous êtes dans le bon dossier
3. Lisez les messages d'erreur dans le terminal
4. Vérifiez votre saisie (types de données corrects)

---

**Bon courage pour votre TP ! 🎓**
