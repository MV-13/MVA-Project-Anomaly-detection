# Deep Learning for Signal Processing – Anomaly Detection

**Authors** : Yanis Allal and Marine Vieillard

---

## Table des matières

1. [Description du projet](#description-du-projet)
2. [Installation](#installation)
3. [Structure du projet](#structure-du-projet)
4. [Partie 1 : Classification des signaux IQ](#partie-1--classification-des-signaux-iq)
5. [Analyse exploratoire des données](#analyse-exploratoire-des-données)
6. [Partie 2 : Détection d'anomalies (OOD)](#partie-2--détection-danomalies-ood)
7. [Résultats](#résultats)
8. [Utilisation rapide](#utilisation-rapide)
---

## Description du projet

Ce projet aborde deux problèmes complémentaires de classification de signaux radio IQ (In-phase/Quadrature) :

1. **Classification supervisée** : Entraîner un réseau de neurones (CNN-LSTM) pour classifier des signaux en 6 types de modulation (C0-C5).

2. **Détection d'anomalies (Out-of-Distribution)** : Détecter les signaux appartenant à des classes inconnues (C6, C7, C8) qui n'étaient pas présentes lors de l'entraînement.

### Défi principal

Les classes inconnues (C6, C7, C8) sont des variantes des classes connues (C3, C4, C5). Elles partagent des caractéristiques très similaires, rendant la détection particulièrement difficile.

---

## Installation

```bash
# Cloner le repository
git clone <repository_url>
cd <repository_name>

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
- PyTorch
- NumPy
- scikit-learn
- h5py
- matplotlib
- seaborn
- tensorboard
- torchmetrics

---

## Structure du projet

```
.
├── Data/
│   ├── train.hdf5           # Données d'entraînement (classes 0-5)
│   ├── validation.hdf5      # Données de validation (classes 0-5)
│   ├── test.hdf5            # Données de test (classes 0-5)
│   ├── test_anomalies.hdf5  # Données de test avec anomalies (classes 0-8)
│   └── samples.hdf5         # Données supplémentaires (classes 0-5)
├── runs/                    # Checkpoints et logs TensorBoard
├── models.py                # Architectures des modèles
├── dataset.py               # Gestion des données
├── train_classifier.py      # Script d'entraînement du classifieur
├── test_classifier.py       # Script de test du classifieur
├── analyze_data.py          # Analyse exploratoire des données
├── anomaly_detector.py      # Détecteur d'anomalies OOD
└── README.md
```

---

## Partie 1 : Classification des signaux IQ

### Architecture du modèle

Nous utilisons un modèle **CNN-LSTM** avec intégration du SNR :

```
Signal IQ [2, 1024] 
    ↓
Conv1D (32 filtres) + BatchNorm + ReLU
    ↓
Conv1D (64 filtres) + BatchNorm + ReLU
    ↓
LSTM Bidirectionnel (hidden=64)
    ↓
Concat avec embedding SNR
    ↓
FC (128) + ReLU
    ↓
FC (6 classes)
```

**Pourquoi cette architecture ?**
- **CNN** : Extrait les patterns locaux du signal (motifs de modulation)
- **LSTM** : Capture les dépendances temporelles
- **Bidirectionnel** : Utilise le contexte passé et futur
- **SNR embedding** : Le SNR est une information cruciale pour la classification

### Entraînement

```bash
python train_classifier.py --name mon_experience
```

**Paramètres par défaut :**
- Batch size : 512
- Epochs : 500
- Optimizer : Adam
- Loss : CrossEntropyLoss
- Augmentation : Rotation de phase aléatoire

**Suivi avec TensorBoard :**
```bash
tensorboard --logdir=runs
```

### Test du classifieur

```bash
python test_classifier.py --checkpoint runs/20251209-160344_run/checkpoint.pt --test Data/test.hdf5
```

**Sorties :**
- Accuracy globale et par SNR
- Matrices de confusion par SNR
- Graphique Accuracy vs SNR

---

## Analyse exploratoire des données

Avant de développer le détecteur d'anomalies, nous avons analysé les données pour comprendre pourquoi les classes inconnues (6-8) ressemblent aux classes connues (3-5).

### Utilisation

```bash
python analyze_data.py \
    --classifier runs/20251209-160344_run/checkpoint.pt \
    --test Data/test_anomalies.hdf5 \
    --save_dir analysis_results
```

### Analyses effectuées

1. **Statistiques des signaux** (`signal_statistics.png`)
   - Amplitude moyenne et écart-type
   - Puissance du signal
   - Variation de phase
   - Corrélation I-Q
   - **Résultat** : Les classes inconnues ont des statistiques quasi-identiques aux classes connues

2. **Analyse fréquentielle** (`frequency_analysis.png`)
   - Spectre moyen par classe
   - Comparaison C3/C5 vs classes inconnues
   - **Résultat** : Les spectres sont très similaires

3. **Visualisation des embeddings** (`tsne_embeddings.png`, `pca_embeddings.png`)
   - Projection t-SNE et PCA des features CNN
   - **Résultat** : Les inconnus forment des clusters **distincts mais proches** des connus

4. **Analyse de confusion** (`unknown_probabilities.png`)
   - Distribution des probabilités softmax pour les classes inconnues
   - Vers quelle classe connue chaque inconnu est classifié

### Insight principal

> Les classes inconnues ne sont pas à la périphérie de l'espace des features, mais forment des clusters légèrement décalés. C'est pourquoi les approches classiques (seuil de confiance, distance au centroïde) échouent, et pourquoi l'approche k-NN fonctionne.

---


## Partie 2 : Détection d'anomalies (OOD)

### Problématique

Le classifieur entraîné sur 6 classes va incorrectement assigner les signaux inconnus (classes 6, 7, 8) à l'une des classes connues. L'objectif est de détecter et rejeter ces signaux inconnus.

### Approche développée

Nous avons développé un détecteur basé sur :

#### 1. Extraction des features CNN

On utilise les activations de la couche avant-dernière du classifieur comme représentation compacte du signal :

**Pourquoi ?** Le CNN a appris une représentation où les classes sont bien séparées. Les signaux similaires ont des features proches.

#### 2. Détection par k-NN multi-échelle

Pour chaque signal de test, on calcule sa distance à ses k plus proches voisins dans les données de validation (connues) :

```python
# Score basé sur le percentile de la distance k-NN
score_k = 1 - percentile(distance_k)
```

**Multi-échelle** : On combine plusieurs valeurs de k pour équilibrer sensibilité et robustesse :
```python
score_knn = 0.4 * score_k3 + 0.35 * score_k5 + 0.25 * score_k10
```

**Pourquoi k-NN ?**
- Les inconnus forment des clusters proches mais distincts des connus
- k-NN capture cette différence locale sans supposer de distribution particulière

#### 3. Combinaison avec la confiance du classifieur

```python
score_combined = 0.7 * score_knn + 0.3 * confidence
```

La confiance (probabilité softmax maximale) apporte une information complémentaire.

#### 4. Seuils adaptatifs par classe

**Observation clé** : Certaines classes sont plus facilement confondues avec les inconnus :
- **C1, C4** : Bien séparées → seuil bas (0.20-0.35)
- **C3, C5** : Confondues avec les variantes → seuil haut (0.45-0.75)

### Utilisation du détecteur

```bash
python anomaly_detector.py \
    --classifier runs/20251209-160344_run/checkpoint.pt \
    --test Data/test_anomalies.hdf5 \
    --val Data/validation.hdf5 \
    --save_dir results_anomaly
```

**Sorties :**
- `results_main.png` : Tableau Pareto, seuils optimaux, matrices de confusion
- `results_distributions.png` : Distributions des scores, courbe ROC, analyse des erreurs
- `optimal_thresholds_6class.npz` : Seuils optimaux sauvegardés

---

## Résultats

### Performance du classifieur (sans anomalies)

| SNR | Accuracy |
|-----|----------|
| 0 dB | 52% |
| 10 dB | 99% |
| 20 dB | 100% |
| 30 dB | 100% |

### Performance du détecteur d'anomalies

| Point de fonctionnement | Recall | Précision | FPR | F1 | Acc Known |
|------------------------|--------|-----------|-----|----|-----------| 
| fpr_15 | 59% | 80% | 14% | 0.68 | 86% |
| fpr_20 | 72% | 78% | 20% | 0.75 | 80% |
| **balanced** | **80%** | **82%** | **23%** | **0.79** | **77%** |
| best_f1 | 88% | 76% | 27% | 0.82 | 73% |

**Seuils optimaux (point balanced) :**
- C0 : 0.25 | C1 : 0.20 | C2 : 0.30 | C3 : 0.75 | C4 : 0.20 | C5 : 0.62

### Choix du point de fonctionnement

- **Application sécurité** (détecter tous les inconnus) : Utiliser `best_f1` ou `recall_90`
- **Application service** (ne pas rejeter de vrais clients) : Utiliser `fpr_15` ou `fpr_20`
- **Équilibré** : Utiliser `balanced`

---

## Utilisation rapide

### 1. Entraîner le classifieur
```bash
python train_classifier.py --name experiment_v1
```

### 2. Tester le classifieur (sans anomalies)
```bash
python test_classifier.py \
    --checkpoint runs/<run_name>/checkpoint.pt \
    --test Data/test.hdf5
```

### 3. Analyser les données (optionnel)

```bash
python analyze_data.py \
    --classifier runs//checkpoint.pt --test Data/test_anomalies.hdf5 --save_dir analysis_results
```

### 4. Détecter les anomalies
```bash
python anomaly_detector.py \
    --classifier runs/<run_name>/checkpoint.pt \
    --test Data/test_anomalies.hdf5 \
    --val Data/validation.hdf5 \
    --save_dir results
```

### 5. Visualiser l'entraînement
```bash
tensorboard --logdir=runs
```

---

## Métriques

| Métrique | Description |
|----------|-------------|
| **Recall** | % d'inconnus correctement détectés |
| **FPR** | % de connus incorrectement rejetés |
| **Precision** | Parmi les rejets, % de vrais inconnus |
| **F1** | Moyenne harmonique Precision/Recall |
| **Acc Known** | 1 - FPR (accuracy sur les classes connues) |

---

## Références

- Architecture CNN-LSTM pour signaux temporels
- Détection OOD par k-NN : [Deep k-Nearest Neighbors](https://arxiv.org/abs/1803.04765)
- Approches de détection d'anomalies : [Deep Learning for Anomaly Detection](https://arxiv.org/abs/2007.02500)

