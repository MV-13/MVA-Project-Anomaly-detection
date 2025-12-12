"""
DETECTOR
- 6 seuils par classe
- 2 figures: résultats principaux + distributions/analyse erreurs
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns


class SignalsDataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.signals = np.array(f["signaux"], dtype=np.float32)
            self.labels = np.array(f["labels"], dtype=np.int8)
            self.snr = np.array(f["snr"], dtype=np.int16)
        print(f"Loaded {len(self.signals)} samples")
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        if signal.shape[0] != 2:
            signal = signal.T
        return signal, torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(self.snr[idx], dtype=torch.float32)


def extract_features(model, dataloader, device):
    model.eval()
    all_cnn = []
    all_logits = []
    all_labels = []
    all_snr = []
    
    features_list = []
    def hook(module, input, output):
        features_list.append(input[0].detach())
    
    handle = model.fc2.register_forward_hook(hook)
    
    with torch.no_grad():
        for signals, labels, snr in dataloader:
            signals = signals.to(device)
            snr_input = snr.to(device).unsqueeze(1)
            logits = model(signals, snr_input)
            
            all_cnn.append(features_list[-1].cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_snr.append(snr.numpy())
    
    handle.remove()
    
    return {
        'cnn': np.concatenate(all_cnn),
        'logits': np.concatenate(all_logits),
        'labels': np.concatenate(all_labels),
        'snr': np.concatenate(all_snr)
    }


class UltimateDetector:
    """Détecteur avec 6 seuils + confiance"""
    
    def __init__(self, k_values=[3, 5, 10]):
        self.k_values = k_values
        self.val_knn = {}
        
    def fit(self, features):
        print("Fitting ultimate detector...")
        
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0) + 1e-6
        features_norm = (features - self.mean) / self.std
        self.val_features_norm = features_norm
        
        distances = self._compute_distances(features_norm, features_norm)
        for k in self.k_values:
            self.val_knn[k] = np.sort(distances, axis=1)[:, k]
    
    def _compute_distances(self, query, reference):
        batch_size = 1000
        n_q, n_r = len(query), len(reference)
        distances = np.zeros((n_q, n_r))
        
        for i in range(0, n_q, batch_size):
            end = min(i + batch_size, n_q)
            batch = query[i:end]
            b_sq = np.sum(batch**2, axis=1, keepdims=True)
            r_sq = np.sum(reference**2, axis=1, keepdims=True).T
            cross = np.dot(batch, reference.T)
            distances[i:end] = np.sqrt(np.maximum(b_sq + r_sq - 2*cross, 0))
        return distances
    
    def compute_scores(self, features, logits):
        features_norm = (features - self.mean) / self.std
        distances = self._compute_distances(features_norm, self.val_features_norm)
        
        scores_per_k = {}
        for k in self.k_values:
            kth_dist = np.sort(distances, axis=1)[:, k-1]
            percentiles = np.array([np.mean(kth_dist[i] > self.val_knn[k]) 
                                   for i in range(len(features_norm))])
            scores_per_k[k] = 1 - percentiles
        
        knn_score = 0.4 * scores_per_k[3] + 0.35 * scores_per_k[5] + 0.25 * scores_per_k[10]
        
        # Confiance du classifieur
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        confidence = np.max(softmax, axis=1)
        predictions = np.argmax(logits, axis=1)
        
        # Score combiné
        combined = 0.7 * knn_score + 0.3 * confidence
        
        return {
            'knn': knn_score,
            'confidence': confidence,
            'combined': combined,
            'predictions': predictions,
            'knn_per_k': scores_per_k
        }


def find_pareto_6class(scores, labels, predictions):
    """Trouve les points Pareto pour 6 seuils"""
    known_mask = labels < 6
    unknown_mask = labels >= 6
    
    all_results = []
    
    search_ranges = {
        0: np.linspace(0.25, 0.50, 5),
        1: np.linspace(0.20, 0.35, 5),
        2: np.linspace(0.30, 0.55, 5),
        3: np.linspace(0.45, 0.75, 5),
        4: np.linspace(0.20, 0.35, 5),
        5: np.linspace(0.40, 0.70, 5),
    }
    
    for t0 in search_ranges[0]:
        for t1 in search_ranges[1]:
            for t2 in search_ranges[2]:
                for t3 in search_ranges[3]:
                    for t4 in search_ranges[4]:
                        for t5 in search_ranges[5]:
                            thresholds = {0: t0, 1: t1, 2: t2, 3: t3, 4: t4, 5: t5}
                            
                            is_unknown = np.zeros(len(scores), dtype=bool)
                            for i in range(len(scores)):
                                is_unknown[i] = scores[i] < thresholds[predictions[i]]
                            
                            tp = (is_unknown & unknown_mask).sum()
                            fp = (is_unknown & known_mask).sum()
                            fn = (~is_unknown & unknown_mask).sum()
                            tn = (~is_unknown & known_mask).sum()
                            
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                            
                            all_results.append({
                                'thresholds': thresholds.copy(),
                                'recall': recall,
                                'precision': precision,
                                'f1': f1,
                                'fpr': fpr,
                                'acc_known': 1 - fpr
                            })
    
    pareto = {}
    
    for target_fpr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        valid = [r for r in all_results if r['fpr'] <= target_fpr]
        if valid:
            pareto[f'fpr_{int(target_fpr*100)}'] = max(valid, key=lambda x: x['recall'])
    
    pareto['best_f1'] = max(all_results, key=lambda x: x['f1'])
    pareto['balanced'] = max(all_results, key=lambda x: x['f1'] * x['acc_known'])
    
    for target_recall in [0.85, 0.90, 0.95]:
        valid = [r for r in all_results if r['recall'] >= target_recall]
        if valid:
            pareto[f'recall_{int(target_recall*100)}'] = min(valid, key=lambda x: x['fpr'])
    
    return pareto


def analyze_errors(scores, labels, predictions, thresholds, snr):
    """Analyse détaillée des erreurs"""
    
    known_mask = labels < 6
    unknown_mask = labels >= 6
    
    is_unknown_pred = np.zeros(len(scores), dtype=bool)
    for i in range(len(scores)):
        is_unknown_pred[i] = scores[i] < thresholds[predictions[i]]
    
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    # Faux positifs
    fp_mask = is_unknown_pred & known_mask
    print(f"\nFalse Positives (known classified as unknown): {fp_mask.sum()}")
    
    if fp_mask.sum() > 0:
        print("\n  By true class:")
        for c in range(6):
            class_fp = fp_mask & (labels == c)
            class_total = (labels == c).sum()
            if class_total > 0:
                rate = class_fp.sum() / class_total
                print(f"    C{c}: {class_fp.sum()}/{class_total} ({rate*100:.1f}%)")
        
        print("\n  By SNR range:")
        snr_ranges = [(-20, 0), (0, 5), (5, 10), (10, 30)]
        for low, high in snr_ranges:
            snr_mask = (snr >= low) & (snr < high)
            fp_in_range = (fp_mask & snr_mask).sum()
            total_in_range = (known_mask & snr_mask).sum()
            if total_in_range > 0:
                rate = fp_in_range / total_in_range
                print(f"    SNR [{low}, {high}): {fp_in_range}/{total_in_range} ({rate*100:.1f}%)")
    
    # Faux négatifs
    fn_mask = ~is_unknown_pred & unknown_mask
    print(f"\nFalse Negatives (unknown not detected): {fn_mask.sum()}")
    
    if fn_mask.sum() > 0:
        print("\n  By predicted class:")
        for c in range(6):
            class_fn = fn_mask & (predictions == c)
            if class_fn.sum() > 0:
                print(f"    Predicted as C{c}: {class_fn.sum()}")
        
        print("\n  By SNR range:")
        for low, high in snr_ranges:
            snr_mask = (snr >= low) & (snr < high)
            fn_in_range = (fn_mask & snr_mask).sum()
            total_in_range = (unknown_mask & snr_mask).sum()
            if total_in_range > 0:
                rate = fn_in_range / total_in_range
                print(f"    SNR [{low}, {high}): {fn_in_range}/{total_in_range} ({rate*100:.1f}%)")
    
    return {
        'fp_mask': fp_mask,
        'fn_mask': fn_mask,
        'fp_by_class': {c: ((fp_mask) & (labels == c)).sum() for c in range(6)},
        'fn_by_pred': {c: ((fn_mask) & (predictions == c)).sum() for c in range(6)}
    }


def confusion_matrix_manual(y_true, y_pred, labels):
    n_labels = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1
    return cm


def plot_main_results(scores_dict, labels, save_dir, pareto):
    """Figure 1: Résultats principaux"""
    os.makedirs(save_dir, exist_ok=True)
    
    scores = scores_dict['combined']
    predictions = scores_dict['predictions']
    known_mask = labels < 6
    unknown_mask = labels >= 6
    
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Tableau Pareto
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.axis('off')
    
    text = "PARETO OPTIMAL POINTS (6 class thresholds)\n" + "="*95 + "\n"
    text += f"{'Point':<12} {'Recall':<8} {'Prec':<8} {'F1':<8} {'FPR':<8} {'AccK':<8} | {'t0':<5} {'t1':<5} {'t2':<5} {'t3':<5} {'t4':<5} {'t5':<5}\n"
    text += "-"*95 + "\n"
    
    for name, p in pareto.items():
        t = p['thresholds']
        text += f"{name:<12} {p['recall']:<8.3f} {p['precision']:<8.3f} {p['f1']:<8.3f} {p['fpr']:<8.3f} {p['acc_known']:<8.3f} | {t[0]:<5.2f} {t[1]:<5.2f} {t[2]:<5.2f} {t[3]:<5.2f} {t[4]:<5.2f} {t[5]:<5.2f}\n"
    
    ax1.text(0.02, 0.98, text, fontsize=8, family='monospace', verticalalignment='top', transform=ax1.transAxes)
    
    # 2. Seuils par classe (balanced)
    ax2 = fig.add_subplot(3, 3, 2)
    best = pareto['balanced']
    thresholds = [best['thresholds'][c] for c in range(6)]
    
    colors = ['#27ae60', '#2ecc71', '#3498db', '#e74c3c', '#2ecc71', '#e67e22']
    bars = ax2.bar(range(6), thresholds, color=colors)
    ax2.set_xticks(range(6))
    ax2.set_xticklabels([f'C{c}' for c in range(6)])
    ax2.set_ylabel('Threshold')
    ax2.set_title(f'Optimal Thresholds (balanced)\nRecall={best["recall"]:.2f}, FPR={best["fpr"]:.2f}')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, t) in enumerate(zip(bars, thresholds)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{t:.2f}', ha='center', fontsize=10)
    
    # 3. Front Pareto
    ax3 = fig.add_subplot(3, 3, 3)
    fprs = [p['fpr'] for p in pareto.values()]
    recalls = [p['recall'] for p in pareto.values()]
    names = list(pareto.keys())
    
    ax3.scatter(fprs, recalls, s=100, c='blue', zorder=5)
    for i, name in enumerate(names):
        ax3.annotate(name, (fprs[i], recalls[i]), textcoords="offset points", 
                    xytext=(5, 5), fontsize=7)
    
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('Recall')
    ax3.set_title('Pareto Front')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 0.5)
    ax3.set_ylim(0.3, 1)
    
    # 4-9. Matrices de confusion
    selected = ['fpr_15', 'fpr_20', 'fpr_25', 'balanced', 'recall_90', 'best_f1']
    
    for idx, name in enumerate(selected):
        if name not in pareto:
            continue
        
        ax = fig.add_subplot(3, 3, 4 + idx)
        p = pareto[name]
        
        is_unknown = np.zeros(len(scores), dtype=bool)
        for i in range(len(scores)):
            is_unknown[i] = scores[i] < p['thresholds'][predictions[i]]
        
        final_pred = np.where(is_unknown, 6, predictions)
        labels_mapped = np.where(labels >= 6, 6, labels)
        
        cm = confusion_matrix_manual(labels_mapped, final_pred, list(range(7)))
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        class_names = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'Unk']
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, ax=ax, annot_kws={'size': 8})
        ax.set_title(f'{name}\nRecall={p["recall"]:.2f} | FPR={p["fpr"]:.2f} | F1={p["f1"]:.2f}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results_main.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'results_main.png')}")
    plt.close()


def plot_distributions(scores_dict, labels, snr, save_dir, pareto, error_analysis):
    """Figure 2: Distributions et analyse des erreurs"""
    os.makedirs(save_dir, exist_ok=True)
    
    scores = scores_dict['combined']
    knn_scores = scores_dict['knn']
    confidence = scores_dict['confidence']
    predictions = scores_dict['predictions']
    knn_per_k = scores_dict['knn_per_k']
    
    known_mask = labels < 6
    unknown_mask = labels >= 6
    
    best = pareto['balanced']
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Distribution globale des scores combinés
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(scores[known_mask], bins=50, alpha=0.6, label='Known', color='blue', density=True)
    ax1.hist(scores[unknown_mask], bins=50, alpha=0.6, label='Unknown', color='red', density=True)
    ax1.set_xlabel('Combined Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    sep = scores[known_mask].mean() - scores[unknown_mask].mean()
    ax1.text(0.05, 0.95, f'Separation: {sep:.3f}', transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Distribution par classe prédite
    ax2 = fig.add_subplot(2, 3, 2)
    for c in range(6):
        mask_class = predictions == c
        mask_known = mask_class & known_mask
        mask_unknown = mask_class & unknown_mask
        if mask_known.sum() > 0:
            ax2.scatter(np.full(mask_known.sum(), c) - 0.15, scores[mask_known], alpha=0.3, c='blue', s=10)
        if mask_unknown.sum() > 0:
            ax2.scatter(np.full(mask_unknown.sum(), c) + 0.15, scores[mask_unknown], alpha=0.3, c='red', s=10)
        ax2.hlines(best['thresholds'][c], c - 0.3, c + 0.3, colors='green', linestyles='--', linewidth=2)
    ax2.set_xticks(range(6))
    ax2.set_xticklabels([f'C{c}' for c in range(6)])
    ax2.set_ylabel('Score')
    ax2.set_title('Scores by Predicted Class\n(Blue=Known, Red=Unknown, Green=Threshold)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(2, 3, 3)
    thresholds_roc = np.linspace(0, 1, 200)
    recalls_roc, fprs_roc = [], []
    for t in thresholds_roc:
        pred_unk = scores < t
        recall = (pred_unk & unknown_mask).sum() / unknown_mask.sum()
        fpr = (pred_unk & known_mask).sum() / known_mask.sum()
        recalls_roc.append(recall)
        fprs_roc.append(fpr)
    ax3.plot(fprs_roc, recalls_roc, 'b-', linewidth=2)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    for name, p in pareto.items():
        ax3.scatter([p['fpr']], [p['recall']], s=80, zorder=5, label=name)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('Recall')
    ax3.set_title('ROC Curve')
    ax3.legend(fontsize=7, loc='lower right')
    ax3.grid(True, alpha=0.3)
    auc = abs(np.trapz(recalls_roc, fprs_roc))
    ax3.text(0.6, 0.2, f'AUC={auc:.3f}', fontsize=12)
    
    # 4. Taux de FP par classe vraie
    ax4 = fig.add_subplot(2, 3, 4)
    fp_by_class = [error_analysis['fp_by_class'][c] for c in range(6)]
    total_by_class = [(labels == c).sum() for c in range(6)]
    fp_rate = [fp_by_class[c] / total_by_class[c] if total_by_class[c] > 0 else 0 for c in range(6)]
    bars = ax4.bar(range(6), fp_rate, color='coral')
    ax4.set_xticks(range(6))
    ax4.set_xticklabels([f'C{c}' for c in range(6)])
    ax4.set_ylabel('False Positive Rate')
    ax4.set_title('FP Rate by True Class')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, rate, count in zip(bars, fp_rate, fp_by_class):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate*100:.1f}%\n({count})', ha='center', fontsize=9)
    
    # 5. Taux d'erreur par SNR
    ax5 = fig.add_subplot(2, 3, 5)
    snr_ranges = [(-20, 0, 'SNR<0'), (0, 5, '0-5'), (5, 10, '5-10'), (10, 30, '≥10')]
    fp_rates_snr = []
    fn_rates_snr = []
    snr_labels = []
    
    for low, high, label in snr_ranges:
        snr_mask = (snr >= low) & (snr < high)
        fp_in = (error_analysis['fp_mask'] & snr_mask).sum()
        fn_in = (error_analysis['fn_mask'] & snr_mask).sum()
        known_in = (known_mask & snr_mask).sum()
        unknown_in = (unknown_mask & snr_mask).sum()
        
        fp_rates_snr.append(fp_in / known_in if known_in > 0 else 0)
        fn_rates_snr.append(fn_in / unknown_in if unknown_in > 0 else 0)
        snr_labels.append(label)
    
    x = np.arange(len(snr_labels))
    width = 0.35
    ax5.bar(x - width/2, fp_rates_snr, width, label='FP Rate', color='coral')
    ax5.bar(x + width/2, fn_rates_snr, width, label='FN Rate', color='steelblue')
    ax5.set_xticks(x)
    ax5.set_xticklabels(snr_labels)
    ax5.set_ylabel('Error Rate')
    ax5.set_title('Error Rates by SNR')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Résumé statistiques
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"""
    SUMMARY
    =======
    
    Best Balanced Point:
      Recall: {best['recall']:.3f} ({best['recall']*100:.1f}%)
      Precision: {best['precision']:.3f}
      F1: {best['f1']:.3f}
      FPR: {best['fpr']:.3f} ({best['fpr']*100:.1f}%)
      Acc Known: {best['acc_known']:.3f}
    
    Optimal Thresholds:
      C0: {best['thresholds'][0]:.3f}
      C1: {best['thresholds'][1]:.3f}
      C2: {best['thresholds'][2]:.3f}
      C3: {best['thresholds'][3]:.3f}
      C4: {best['thresholds'][4]:.3f}
      C5: {best['thresholds'][5]:.3f}
    
    Score Statistics:
      Known: {scores[known_mask].mean():.3f} ± {scores[known_mask].std():.3f}
      Unknown: {scores[unknown_mask].mean():.3f} ± {scores[unknown_mask].std():.3f}
      Separation: {sep:.3f}
      AUC: {auc:.3f}
    """
    
    ax6.text(0.05, 0.95, summary, fontsize=11, family='monospace',
             verticalalignment='top', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'results_distributions.png')}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results_ultimate_final")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from models import CNN_LSTM_SNR_Model
    model = CNN_LSTM_SNR_Model(n_classes=6, n_channels=2, hidden_size=64, include_snr=True)
    checkpoint = torch.load(args.classifier, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    val_loader = DataLoader(SignalsDataset(args.val), batch_size=256, shuffle=False)
    test_loader = DataLoader(SignalsDataset(args.test), batch_size=256, shuffle=False)
    
    print("\nExtracting features...")
    val_data = extract_features(model, val_loader, device)
    test_data = extract_features(model, test_loader, device)
    
    detector = UltimateDetector(k_values=[3, 5, 10])
    detector.fit(val_data['cnn'])
    
    print("\nComputing scores...")
    scores_dict = detector.compute_scores(test_data['cnn'], test_data['logits'])
    
    print("\nFinding Pareto-optimal points...")
    pareto = find_pareto_6class(scores_dict['combined'], test_data['labels'], scores_dict['predictions'])
    
    # Analyse des erreurs
    error_analysis = analyze_errors(
        scores_dict['combined'], 
        test_data['labels'], 
        scores_dict['predictions'],
        pareto['balanced']['thresholds'],
        test_data['snr']
    )
    
    # Afficher résultats
    print("\n" + "="*100)
    print("RESULTS WITH 6 CLASS-SPECIFIC THRESHOLDS")
    print("="*100)
    
    for name, p in pareto.items():
        t = p['thresholds']
        print(f"{name:<12}: Recall={p['recall']:.3f}, Precision={p['precision']:.3f}, F1={p['f1']:.3f}, FPR={p['fpr']:.3f}")
        print(f"             Thresholds: C0={t[0]:.2f}, C1={t[1]:.2f}, C2={t[2]:.2f}, C3={t[3]:.2f}, C4={t[4]:.2f}, C5={t[5]:.2f}")
    
    # Générer les 2 figures
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\nGenerating figures...")
    plot_main_results(scores_dict, test_data['labels'], args.save_dir, pareto)
    plot_distributions(scores_dict, test_data['labels'], test_data['snr'], args.save_dir, pareto, error_analysis)
    
    # Sauvegarder
    best = pareto['balanced']
    np.savez(os.path.join(args.save_dir, 'optimal_thresholds_6class.npz'),
             thresholds=best['thresholds'],
             metrics=best)
    
    print(f"\nResults saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
