import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from dataset import SignalsDataset
import os
from datetime import datetime

# Configuration
test_path = 'Data/test_anomalies.hdf5'
model_path = 'best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Créer le dossier pour sauvegarder les figures
output_dir = 'test_results'
os.makedirs(output_dir, exist_ok=True)

print(f"Utilisation de: {device}")
print(f"Résultats sauvegardés dans: {output_dir}/")


# Définir le modèle (même architecture que dans train.py)
class SimpleSignalClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        logits = self.classifier(x)
        return logits


def get_predictions(model, dataloader, device):
    """Extrait toutes les prédictions et labels"""
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    all_probs = []
    
    with torch.no_grad():
        for signals, labels, snr in dataloader:
            signals = signals.to(device)
            
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = probs.max(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(max_probs.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confidences),
        np.array(all_probs)
    )


def find_best_threshold(confidences, labels):
    """Trouve le meilleur seuil pour la détection d'anomalies"""
    true_binary = (labels >= 6).astype(int)
    thresholds = np.linspace(0.3, 0.95, 50)
    best_f1 = 0
    best_threshold = 0.5
    
    results = []
    for thresh in thresholds:
        pred_binary = (confidences < thresh).astype(int)
        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        results.append((thresh, precision, recall, f1))
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1, results


def plot_class_distribution(labels, output_dir):
    """Visualise la distribution des classes"""
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = ['green' if c < 6 else 'red' for c in unique]
    ax.bar(unique, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Classe', fontsize=12)
    ax.set_ylabel('Nombre d\'échantillons', fontsize=12)
    ax.set_title('Distribution des classes dans le dataset de test', fontsize=14, fontweight='bold')
    ax.set_xticks(unique)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Classes normales (0-5)'),
        Patch(facecolor='red', alpha=0.7, label='Classes anomalies (6-8)')
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== Distribution des classes ===")
    for c, count in zip(unique, counts):
        status = "NORMALE" if c < 6 else "ANOMALIE"
        print(f"  Classe {c} ({status}): {count} échantillons")


def plot_confusion_matrix_multiclass(labels, preds, output_dir):
    """Matrice de confusion pour les classes normales (0-5)"""
    mask_normal = labels < 6
    labels_normal = labels[mask_normal]
    preds_normal = preds[mask_normal]
    
    cm = confusion_matrix(labels_normal, preds_normal, labels=range(6))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(6), yticklabels=range(6), 
                ax=ax, cbar_kws={'label': 'Nombre'})
    ax.set_xlabel('Classe prédite', fontsize=12)
    ax.set_ylabel('Classe réelle', fontsize=12)
    ax.set_title('Matrice de confusion - Classes normales (0-5)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_confusion_matrix_multiclass.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== Accuracy par classe normale ===")
    for i in range(6):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            print(f"  Classe {i}: {acc:.2f}% ({cm[i, i]}/{cm[i].sum()})")
    
    return cm


def plot_metrics_per_class(labels, preds, output_dir):
    """Calcule et affiche les métriques par classe"""
    mask_normal = labels < 6
    labels_normal = labels[mask_normal]
    preds_normal = preds[mask_normal]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_normal, preds_normal, labels=range(6), zero_division=0
    )
    
    # Affichage textuel
    print("\n=== Métriques par classe (0-5) ===")
    print(f"{'Classe':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i in range(6):
        print(f"{i:<8} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    print("-" * 60)
    print(f"{'Moyenne':<8} {precision.mean():<12.4f} {recall.mean():<12.4f} {f1.mean():<12.4f} {support.sum():<10}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].bar(range(6), precision, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Classe', fontsize=11)
    axes[0].set_ylabel('Precision', fontsize=11)
    axes[0].set_title('Precision par classe', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(6))
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(precision.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Moy={precision.mean():.3f}')
    axes[0].legend()
    
    axes[1].bar(range(6), recall, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Classe', fontsize=11)
    axes[1].set_ylabel('Recall', fontsize=11)
    axes[1].set_title('Recall par classe', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(6))
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(recall.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Moy={recall.mean():.3f}')
    axes[1].legend()
    
    axes[2].bar(range(6), f1, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Classe', fontsize=11)
    axes[2].set_ylabel('F1-Score', fontsize=11)
    axes[2].set_title('F1-Score par classe', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(6))
    axes[2].set_ylim([0, 1.05])
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].axhline(f1.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Moy={f1.mean():.3f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_metrics_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return precision, recall, f1


def plot_threshold_analysis(confidences, labels, output_dir):
    """Analyse de l'impact du seuil"""
    best_threshold, best_f1, results = find_best_threshold(confidences, labels)
    
    thresholds_list, precisions, recalls, f1s = zip(*results)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(thresholds_list, precisions, label='Precision', marker='o', markersize=4, linewidth=2)
    ax.plot(thresholds_list, recalls, label='Recall', marker='s', markersize=4, linewidth=2)
    ax.plot(thresholds_list, f1s, label='F1-Score', marker='^', markersize=4, linewidth=2)
    ax.axvline(best_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Meilleur seuil = {best_threshold:.3f}')
    ax.set_xlabel('Seuil de confiance', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Impact du seuil sur les métriques de détection d\'anomalies', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Meilleur seuil ===")
    print(f"  Seuil optimal: {best_threshold:.4f}")
    print(f"  F1-Score: {best_f1:.4f}")
    
    return best_threshold


def plot_confusion_matrix_binary(labels, confidences, threshold, output_dir):
    """Matrice de confusion binaire (Normal vs Anomalie)"""
    pred_binary = (confidences < threshold).astype(int)
    true_binary = (labels >= 6).astype(int)
    
    cm = confusion_matrix(true_binary, pred_binary)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', 
                xticklabels=['Normal', 'Anomalie'],
                yticklabels=['Normal', 'Anomalie'],
                ax=ax, cbar_kws={'label': 'Nombre'}, annot_kws={'size': 16})
    ax.set_xlabel('Prédiction', fontsize=12)
    ax.set_ylabel('Vérité terrain', fontsize=12)
    ax.set_title(f'Matrice de confusion - Détection d\'anomalies\n(seuil={threshold:.3f})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_confusion_matrix_binary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Métriques
    precision_binary = precision_score(true_binary, pred_binary)
    recall_binary = recall_score(true_binary, pred_binary)
    f1_binary = f1_score(true_binary, pred_binary)
    
    print(f"\n=== Détection d'anomalies (seuil={threshold:.4f}) ===")
    print(f"  Precision: {precision_binary:.4f}")
    print(f"  Recall: {recall_binary:.4f}")
    print(f"  F1-Score: {f1_binary:.4f}")
    print(f"\n  Vrais Négatifs (TN): {cm[0,0]}")
    print(f"  Faux Positifs (FP): {cm[0,1]}")
    print(f"  Faux Négatifs (FN): {cm[1,0]}")
    print(f"  Vrais Positifs (TP): {cm[1,1]}")
    
    return cm, precision_binary, recall_binary, f1_binary


def plot_confidence_histograms(labels, confidences, threshold, output_dir):
    """Histogrammes de confiance par classe"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for class_id in range(9):
        mask = labels == class_id
        if mask.sum() > 0:
            class_confidences = confidences[mask]
            is_anomaly = class_id >= 6
            color = 'red' if is_anomaly else 'green'
            status = "ANOMALIE" if is_anomaly else "NORMALE"
            
            axes[class_id].hist(class_confidences, bins=30, color=color, alpha=0.6, edgecolor='black')
            axes[class_id].axvline(threshold, color='blue', linestyle='--', linewidth=2, 
                                   label=f'Seuil={threshold:.3f}')
            axes[class_id].set_xlabel('Confiance', fontsize=10)
            axes[class_id].set_ylabel('Fréquence', fontsize=10)
            axes[class_id].set_title(f'Classe {class_id} ({status})\nμ={class_confidences.mean():.3f}, σ={class_confidences.std():.3f}',
                                     fontsize=11, fontweight='bold')
            axes[class_id].legend(fontsize=8)
            axes[class_id].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_confidence_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== Statistiques de confiance par classe ===")
    for class_id in range(9):
        mask = labels == class_id
        if mask.sum() > 0:
            class_confidences = confidences[mask]
            status = "ANOMALIE" if class_id >= 6 else "NORMALE"
            print(f"  Classe {class_id} ({status}): μ={class_confidences.mean():.4f}, σ={class_confidences.std():.4f}, "
                  f"min={class_confidences.min():.4f}, max={class_confidences.max():.4f}")


def plot_confidence_boxplots(labels, confidences, threshold, output_dir):
    """Box plots des confidences par classe"""
    confidence_data = []
    class_labels_list = []
    
    for class_id in range(9):
        mask = labels == class_id
        if mask.sum() > 0:
            confidence_data.append(confidences[mask])
            class_labels_list.append(f'C{class_id}')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bp = ax.boxplot(confidence_data, labels=class_labels_list, patch_artist=True, widths=0.6)
    
    # Colorer les boxes
    colors = ['green' if int(label[1:]) < 6 else 'red' for label in class_labels_list]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.axhline(threshold, color='blue', linestyle='--', linewidth=2, 
               label=f'Seuil optimal = {threshold:.3f}')
    ax.set_xlabel('Classe', fontsize=12)
    ax.set_ylabel('Confiance', fontsize=12)
    ax.set_title('Distribution des confidences par classe', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_confidence_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_all_classes(labels, preds, output_dir):
    """Matrice de confusion pour toutes les classes (0-8)"""
    cm_all = confusion_matrix(labels, preds, labels=range(9))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'C{i}' for i in range(9)], 
                yticklabels=[f'C{i}' for i in range(9)], 
                ax=ax, cbar_kws={'label': 'Nombre'})
    ax.set_xlabel('Classe prédite', fontsize=12)
    ax.set_ylabel('Classe réelle', fontsize=12)
    ax.set_title('Matrice de confusion - Toutes les classes (0-8)', fontsize=14, fontweight='bold')
    
    # Ajouter des lignes pour séparer normales et anomalies
    ax.axhline(6, color='red', linewidth=3)
    ax.axvline(6, color='red', linewidth=3)
    
    # Ajouter des annotations
    ax.text(3, -0.8, 'Classes normales', ha='center', fontsize=12, fontweight='bold', color='green')
    ax.text(7.5, -0.8, 'Anomalies', ha='center', fontsize=12, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_confusion_matrix_all_classes.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_errors(labels, confidences, pred_binary, true_binary, output_dir):
    """Analyse détaillée des erreurs"""
    print("\n=== Analyse des erreurs ===")
    
    # Analyse des classes anomalies
    print("\n--- Détection par classe anomalie ---")
    for class_id in [6, 7, 8]:
        mask = labels == class_id
        if mask.sum() > 0:
            class_pred_binary = pred_binary[mask]
            class_confidences = confidences[mask]
            
            detected = class_pred_binary.sum()
            total = len(class_pred_binary)
            detection_rate = detected / total * 100
            
            print(f"\nClasse {class_id}:")
            print(f"  Total: {total} échantillons")
            print(f"  Détectés: {detected} ({detection_rate:.2f}%)")
            print(f"  Non détectés: {total - detected} ({100-detection_rate:.2f}%)")
            print(f"  Confiance: μ={class_confidences.mean():.4f}, σ={class_confidences.std():.4f}")
    
    # Faux positifs
    print("\n--- Faux Positifs (normales → anomalies) ---")
    fp_mask = (true_binary == 0) & (pred_binary == 1)
    if fp_mask.sum() > 0:
        fp_labels = labels[fp_mask]
        unique_fp, counts_fp = np.unique(fp_labels, return_counts=True)
        for c, count in zip(unique_fp, counts_fp):
            total_in_class = (labels == c).sum()
            print(f"  Classe {c}: {count}/{total_in_class} ({count/total_in_class*100:.2f}%)")
    else:
        print("  Aucun faux positif!")
    
    # Faux négatifs
    print("\n--- Faux Négatifs (anomalies → normales) ---")
    fn_mask = (true_binary == 1) & (pred_binary == 0)
    if fn_mask.sum() > 0:
        fn_labels = labels[fn_mask]
        unique_fn, counts_fn = np.unique(fn_labels, return_counts=True)
        for c, count in zip(unique_fn, counts_fn):
            total_in_class = (labels == c).sum()
            print(f"  Classe {c}: {count}/{total_in_class} ({count/total_in_class*100:.2f}%)")
    else:
        print("  Aucun faux négatif!")


def main():
    print("="*70)
    print("ÉVALUATION DU MODÈLE DE DÉTECTION D'ANOMALIES")
    print("="*70)
    
    # Charger le modèle
    print("\n[1/9] Chargement du modèle...")
    model = SimpleSignalClassifier(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("  ✓ Modèle chargé")
    
    # Charger les données
    print("\n[2/9] Chargement des données de test...")
    test_dataset = SignalsDataset(
        path_to_data=test_path,
        transform=None,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    print(f"  ✓ Dataset chargé: {len(test_dataset)} échantillons")
    
    # Extraire les prédictions
    print("\n[3/9] Extraction des prédictions...")
    preds, labels, confidences, probs = get_predictions(model, test_loader, device)
    print(f"  ✓ Prédictions extraites")
    
    # Visualisations
    print("\n[4/9] Distribution des classes...")
    plot_class_distribution(labels, output_dir)
    
    print("\n[5/9] Matrice de confusion multi-classe...")
    cm_multi = plot_confusion_matrix_multiclass(labels, preds, output_dir)
    
    print("\n[6/9] Métriques par classe...")
    precision_pc, recall_pc, f1_pc = plot_metrics_per_class(labels, preds, output_dir)
    
    print("\n[7/9] Analyse du seuil optimal...")
    best_threshold = plot_threshold_analysis(confidences, labels, output_dir)
    
    print("\n[8/9] Détection d'anomalies...")
    pred_binary = (confidences < best_threshold).astype(int)
    true_binary = (labels >= 6).astype(int)
    cm_binary, prec_bin, rec_bin, f1_bin = plot_confusion_matrix_binary(
        labels, confidences, best_threshold, output_dir
    )
    
    print("\n[9/9] Visualisations complémentaires...")
    plot_confidence_histograms(labels, confidences, best_threshold, output_dir)
    plot_confidence_boxplots(labels, confidences, best_threshold, output_dir)
    plot_confusion_matrix_all_classes(labels, preds, output_dir)
    
    # Analyse des erreurs
    analyze_errors(labels, confidences, pred_binary, true_binary, output_dir)
    
    print("\n" + "="*70)
    print("RÉSUMÉ FINAL")
    print("="*70)
    print(f"\nClasses normales (0-5):")
    print(f"  Precision moyenne: {precision_pc.mean():.4f}")
    print(f"  Recall moyen: {recall_pc.mean():.4f}")
    print(f"  F1-Score moyen: {f1_pc.mean():.4f}")
    print(f"\nDétection d'anomalies:")
    print(f"  Seuil optimal: {best_threshold:.4f}")
    print(f"  Precision: {prec_bin:.4f}")
    print(f"  Recall: {rec_bin:.4f}")
    print(f"  F1-Score: {f1_bin:.4f}")
    print(f"\n✓ Toutes les visualisations sont sauvegardées dans: {output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()