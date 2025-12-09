import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SignalsDataset
from models import CNN_LSTM_SNR_Model
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist


def extract_features_and_logits(model, dataloader, device):
    """Extrait features et logits"""
    model.eval()
    all_features = []
    all_logits = []
    all_labels = []
    all_snr = []
    
    features_hook = []
    
    def hook_fn(module, input, output):
        features_hook.append(input[0].detach())
    
    hook = model.fc2.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for signals, labels, snr in dataloader:
            signals = signals.to(device)
            snr_input = snr.to(device).unsqueeze(1)
            
            outputs = model(signals, snr_input)
            
            all_logits.append(outputs.cpu())
            all_labels.append(labels)
            all_snr.append(snr)
            
            if features_hook:
                all_features.append(features_hook[-1].cpu())
    
    hook.remove()
    
    features = torch.cat(all_features, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    snr_values = torch.cat(all_snr, dim=0).numpy()
    
    return features, logits, labels, snr_values


def method_knn_distance(test_features, train_features, k=10, metric='euclidean'):
    """
    Distance KNN avec plusieurs variantes.
    Retourne: score normalisé (plus haut = plus confiant)
    """
    distances = cdist(test_features, train_features, metric=metric)
    
    # k plus proches voisins
    knn_distances = np.partition(distances, k, axis=1)[:, :k]
    
    # Moyenne des distances aux k voisins
    mean_knn_dist = np.mean(knn_distances, axis=1)
    
    # Normaliser et inverser
    max_dist = np.percentile(mean_knn_dist, 99)  # Éviter les outliers
    normalized = np.clip(mean_knn_dist / max_dist, 0, 1)
    
    return 1 - normalized


def method_knn_weighted(test_features, train_features, k=10, metric='euclidean'):
    """
    KNN avec pondération par distance (les plus proches comptent plus).
    """
    distances = cdist(test_features, train_features, metric=metric)
    
    knn_distances = np.partition(distances, k, axis=1)[:, :k]
    
    # Pondération: 1/distance (les proches pèsent plus)
    weights = 1.0 / (knn_distances + 1e-6)
    weighted_dist = np.sum(knn_distances * weights, axis=1) / np.sum(weights, axis=1)
    
    # Normaliser
    max_dist = np.percentile(weighted_dist, 99)
    normalized = np.clip(weighted_dist / max_dist, 0, 1)
    
    return 1 - normalized


def method_knn_class_aware(test_features, train_features, train_labels, k=10, n_classes=6):
    """
    KNN avec conscience des classes: distance minimale aux k voisins de CHAQUE classe.
    """
    all_scores = []
    
    for c in range(n_classes):
        class_features = train_features[train_labels == c]
        
        if len(class_features) == 0:
            continue
        
        distances = cdist(test_features, class_features, metric='euclidean')
        k_actual = min(k, len(class_features))
        
        knn_distances = np.partition(distances, k_actual-1, axis=1)[:, :k_actual]
        mean_dist = np.mean(knn_distances, axis=1)
        
        all_scores.append(mean_dist)
    
    # Distance minimale à n'importe quelle classe
    min_distances = np.min(np.array(all_scores), axis=0)
    
    # Normaliser
    max_dist = np.percentile(min_distances, 99)
    normalized = np.clip(min_distances / max_dist, 0, 1)
    
    return 1 - normalized


def method_energy_score(logits, temperature=1.0):
    """Energy score"""
    energy = -temperature * np.log(np.sum(np.exp(logits / temperature), axis=1))
    return -energy


def method_max_logit(logits):
    """Maximum logit"""
    return np.max(logits, axis=1)


def compute_detection_score(logits, features, val_features, val_labels, method='knn', k=10):
    """
    Calcule le score de détection.
    """
    if method == 'knn':
        return method_knn_distance(features, val_features, k=k)
    
    elif method == 'knn_weighted':
        return method_knn_weighted(features, val_features, k=k)
    
    elif method == 'knn_class':
        return method_knn_class_aware(features, val_features, val_labels, k=k)
    
    elif method == 'energy':
        return method_energy_score(logits)
    
    elif method == 'max_logit':
        return method_max_logit(logits)
    
    elif method == 'hybrid_knn_energy':
        # Combinaison KNN + Energy
        score_knn = method_knn_distance(features, val_features, k=k)
        score_energy = method_energy_score(logits)
        
        # Normaliser
        score_knn = (score_knn - score_knn.min()) / (score_knn.max() - score_knn.min() + 1e-10)
        score_energy = (score_energy - score_energy.min()) / (score_energy.max() - score_energy.min() + 1e-10)
        
        return 0.7 * score_knn + 0.3 * score_energy
    
    elif method == 'hybrid_knn_logit':
        # Combinaison KNN + Max Logit
        score_knn = method_knn_distance(features, val_features, k=k)
        score_logit = method_max_logit(logits)
        
        score_knn = (score_knn - score_knn.min()) / (score_knn.max() - score_knn.min() + 1e-10)
        score_logit = (score_logit - score_logit.min()) / (score_logit.max() - score_logit.min() + 1e-10)
        
        return 0.7 * score_knn + 0.3 * score_logit
    
    elif method == 'ensemble_all':
        # Combinaison de tout
        score_knn = method_knn_class_aware(features, val_features, val_labels, k=k)
        score_energy = method_energy_score(logits)
        score_logit = method_max_logit(logits)
        
        # Normaliser
        score_knn = (score_knn - score_knn.min()) / (score_knn.max() - score_knn.min() + 1e-10)
        score_energy = (score_energy - score_energy.min()) / (score_energy.max() - score_energy.min() + 1e-10)
        score_logit = (score_logit - score_logit.min()) / (score_logit.max() - score_logit.min() + 1e-10)
        
        return 0.5 * score_knn + 0.25 * score_energy + 0.25 * score_logit
    
    else:
        raise ValueError(f"Unknown method: {method}")


def find_optimal_threshold(val_scores, strictness='medium'):
    """Trouve le seuil optimal"""
    strictness_params = {
        'very_low': 0.1,
        'low': 0.5,
        'medium': 1.0,
        'high': 2.0,
        'very_high': 5.0,
        'extreme': 10.0
    }
    
    percentile = strictness_params.get(strictness, 1.0)
    threshold = np.percentile(val_scores, percentile)
    
    return threshold


def predict_with_unknown(checkpoint_path, test_path, batch_size=512,
                        strictness='medium', method='knn', k=10,
                        magnitude_only=True, window_size=256, include_snr=True):
    """Prédictions avec détection"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Datasets
    print("Loading datasets...")
    val_dataset = SignalsDataset("Data/validation.hdf5", transform=None, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = SignalsDataset(test_path, transform=None, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Modèle
    print("Loading model...")
    model = CNN_LSTM_SNR_Model(n_classes=6, n_channels=2, hidden_size=64, include_snr=include_snr)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    # Extraction
    print("\nExtracting features...")
    val_features, val_logits, val_labels, _ = extract_features_and_logits(model, val_loader, device)
    test_features, test_logits, test_labels, test_snr = extract_features_and_logits(model, test_loader, device)
    
    # Scores
    print(f"Computing scores (method: {method}, k: {k})...")
    val_scores = compute_detection_score(val_logits, val_features, val_features, val_labels, method=method, k=k)
    test_scores = compute_detection_score(test_logits, test_features, val_features, val_labels, method=method, k=k)
    
    # Seuil
    threshold = find_optimal_threshold(val_scores, strictness)
    
    # Stats
    known_mask = test_labels < 6
    unknown_mask = test_labels >= 6
    
    print(f"\n{'='*70}")
    print(f"METHOD: {method} | K: {k} | STRICTNESS: {strictness}")
    print(f"{'='*70}")
    
    print(f"\nValidation (known):")
    print(f"  Mean: {val_scores.mean():.4f} | Std: {val_scores.std():.4f}")
    print(f"  Range: [{val_scores.min():.4f}, {val_scores.max():.4f}]")
    
    if known_mask.any():
        known_scores = test_scores[known_mask]
        print(f"\nTest Known (0-5):")
        print(f"  Mean: {known_scores.mean():.4f} | Std: {known_scores.std():.4f}")
        print(f"  Range: [{known_scores.min():.4f}, {known_scores.max():.4f}]")
    
    if unknown_mask.any():
        unknown_scores = test_scores[unknown_mask]
        print(f"\nTest Unknown (6-8):")
        print(f"  Mean: {unknown_scores.mean():.4f} | Std: {unknown_scores.std():.4f}")
        print(f"  Range: [{unknown_scores.min():.4f}, {unknown_scores.max():.4f}]")
        
        # Séparation des distributions
        separation = abs(known_scores.mean() - unknown_scores.mean()) / (known_scores.std() + unknown_scores.std())
        print(f"\n  📊 Distribution Separation: {separation:.3f}")
    
    print(f"\n{'='*70}")
    print(f"THRESHOLD: {threshold:.4f}")
    print(f"{'='*70}")
    
    if known_mask.any() and unknown_mask.any():
        pct_val = (val_scores < threshold).mean() * 100
        pct_known = (test_scores[known_mask] < threshold).mean() * 100
        pct_unknown = (test_scores[unknown_mask] < threshold).mean() * 100
        
        print(f"Val < threshold:     {pct_val:.2f}%")
        print(f"Known < threshold:   {pct_known:.2f}%")
        print(f"Unknown < threshold: {pct_unknown:.2f}%")
    
    # Prédictions
    probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    base_predictions = np.argmax(probs, axis=1)
    
    is_known = test_scores >= threshold
    final_predictions = np.where(is_known, base_predictions, 6)
    
    # Métriques
    n_total = len(final_predictions)
    n_true_unknown = unknown_mask.sum()
    n_detected_unknown = (final_predictions == 6).sum()
    n_correct_unknown = ((final_predictions == 6) & unknown_mask).sum()
    n_false_unknown = ((final_predictions == 6) & known_mask).sum()
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Total: {n_total} | True Unknown: {n_true_unknown} | Detected: {n_detected_unknown}")
    print(f"  ✓ True Positives: {n_correct_unknown}")
    print(f"  ✗ False Positives: {n_false_unknown}")
    
    if n_true_unknown > 0:
        recall = n_correct_unknown / n_true_unknown
        print(f"\n📊 RECALL: {recall:.4f} ({recall*100:.1f}%)")
    else:
        recall = 0
    
    if n_detected_unknown > 0:
        precision = n_correct_unknown / n_detected_unknown
        print(f"📊 PRECISION: {precision:.4f} ({precision*100:.1f}%)")
    else:
        precision = 0
    
    if recall > 0 and precision > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"📊 F1-SCORE: {f1:.4f} ({f1*100:.1f}%)")
    
    known_kept = known_mask & (final_predictions != 6)
    if known_kept.sum() > 0:
        acc_known = (final_predictions[known_kept] == test_labels[known_kept]).mean()
        print(f"📊 ACCURACY (Known): {acc_known:.4f} ({acc_known*100:.1f}%)")
    
    print(f"{'='*70}\n")
    
    return final_predictions, probs, test_labels, test_snr, test_scores, threshold


def plot_results(predictions, true_labels, test_scores, threshold, method, k, strictness, save_dir):
    """Affiche distribution ET matrice de confusion"""
    
    # Séparer scores
    known_mask = true_labels < 6
    unknown_mask = true_labels >= 6
    
    val_scores = test_scores[known_mask][:5000]  # Simuler val
    known_scores = test_scores[known_mask]
    unknown_scores = test_scores[unknown_mask]
    
    # Figure avec 3 subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Histogramme
    ax1 = fig.add_subplot(gs[0, :])
    bins = 60
    ax1.hist(val_scores, bins=bins, alpha=0.4, label='Val (Known)', color='green', density=True)
    ax1.hist(known_scores, bins=bins, alpha=0.5, label='Test Known (0-5)', color='blue', density=True)
    ax1.hist(unknown_scores, bins=bins, alpha=0.5, label='Test Unknown (6-8)', color='red', density=True)
    ax1.axvline(threshold, color='black', linestyle='--', linewidth=2.5, 
                label=f'Threshold: {threshold:.4f}')
    
    ax1.set_xlabel('Detection Score', fontsize=13)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.set_title(f'Score Distribution (Method: {method}, K: {k}, Strictness: {strictness})', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box Plot
    ax2 = fig.add_subplot(gs[1, 0])
    data = [val_scores, known_scores, unknown_scores]
    labels = ['Val\n(Known)', 'Test\n(Known)', 'Test\n(Unknown)']
    bp = ax2.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.axhline(threshold, color='black', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold:.4f}')
    ax2.set_ylabel('Detection Score', fontsize=13)
    ax2.set_title('Score Distribution (Box Plot)', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[1, 1])
    
    true_labels_mapped = np.where(true_labels >= 6, 6, true_labels)
    class_names = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'Unk']
    
    cm = confusion_matrix(true_labels_mapped, predictions, labels=list(range(7)))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1, ax=ax3)
    
    ax3.set_title('Confusion Matrix', fontsize=15, fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('True', fontsize=12)
    
    # Sauvegarder
    save_path = os.path.join(save_dir, f'results_{method}_k{k}_{strictness}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def compare_k_values(checkpoint_path, test_path, method='knn', strictness='high', k_values=[2, 5, 10, 20, 50]):
    """Compare plusieurs valeurs de k"""
    print(f"\n{'='*70}")
    print(f"COMPARING K VALUES: {k_values}")
    print(f"{'='*70}\n")
    
    results = []
    
    for k in k_values:
        print(f"\n{'#'*70}")
        print(f"Testing K = {k}")
        print(f"{'#'*70}")
        
        predictions, probs, labels, snr, scores, threshold = predict_with_unknown(
            checkpoint_path=checkpoint_path,
            test_path=test_path,
            strictness=strictness,
            method=method,
            k=k
        )
        
        # Métriques
        unknown_mask = labels >= 6
        n_true_unknown = unknown_mask.sum()
        n_correct = ((predictions == 6) & unknown_mask).sum()
        
        recall = n_correct / n_true_unknown if n_true_unknown > 0 else 0
        
        results.append({
            'k': k,
            'recall': recall,
            'threshold': threshold,
            'scores': scores,
            'predictions': predictions,
            'labels': labels
        })
    
    # Résumé
    print(f"\n{'='*70}")
    print(f"SUMMARY - K COMPARISON")
    print(f"{'='*70}")
    print(f"{'K':<5} | {'Recall':<10} | {'Threshold':<12}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['k']:<5} | {r['recall']:<10.4f} | {r['threshold']:<12.6f}")
    
    # Meilleur k
    best = max(results, key=lambda x: x['recall'])
    print(f"\n✓ Best K: {best['k']} (Recall: {best['recall']:.4f})")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Unknown detection with KNN optimization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test", type=str, default="Data/test_with_unknown.hdf5")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--strictness", type=str, default='high',
                       choices=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'])
    parser.add_argument("--method", type=str, default='knn',
                       choices=['knn', 'knn_weighted', 'knn_class', 'energy', 'max_logit', 
                               'hybrid_knn_energy', 'hybrid_knn_logit', 'ensemble_all'])
    parser.add_argument("--k", type=int, default=20, help="K for KNN methods")
    parser.add_argument("--compare_k", action='store_true', help="Compare multiple K values")
    parser.add_argument("--save_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.compare_k:
        # Comparer plusieurs K
        k_values = [2, 5, 10, 20, 50, 100]
        compare_k_values(args.checkpoint, args.test, args.method, args.strictness, k_values)
    else:
        # Test simple
        predictions, probs, labels, snr, scores, threshold = predict_with_unknown(
            checkpoint_path=args.checkpoint,
            test_path=args.test,
            batch_size=args.batch_size,
            strictness=args.strictness,
            method=args.method,
            k=args.k
        )
        
        # Afficher résultats
        plot_results(predictions, labels, scores, threshold, 
                    args.method, args.k, args.strictness, args.save_dir)
        
        # Sauvegarder
        results_path = os.path.join(args.save_dir, f'results_{args.method}_k{args.k}_{args.strictness}.npz')
        np.savez(results_path, predictions=predictions, probabilities=probs, 
                true_labels=labels, snr_values=snr, scores=scores, threshold=threshold)
        print(f"✓ Saved: {results_path}\n")


if __name__ == "__main__":
    main()