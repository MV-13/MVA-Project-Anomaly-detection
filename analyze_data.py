"""
ANALYSE DES DONNÉES
Comprendre pourquoi les classes 6-8 ressemblent tant aux classes 3 et 5.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import h5py
import os

# Import du modèle
import sys
sys.path.append('.')


class SimpleDataset(Dataset):
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


def analyze_signal_statistics(test_path, save_dir):
    """Analyse les statistiques des signaux par classe"""
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = SimpleDataset(test_path)
    
    # Statistiques par classe
    stats_by_class = {}
    
    for idx in range(len(dataset)):
        signal, label, snr = dataset[idx]
        label = label.item()
        
        if label not in stats_by_class:
            stats_by_class[label] = {
                'amplitude_mean': [],
                'amplitude_std': [],
                'power': [],
                'phase_std': [],
                'iq_correlation': [],
                'snr': []
            }
        
        I, Q = signal[0].numpy(), signal[1].numpy()
        amplitude = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)
        
        stats_by_class[label]['amplitude_mean'].append(np.mean(amplitude))
        stats_by_class[label]['amplitude_std'].append(np.std(amplitude))
        stats_by_class[label]['power'].append(np.mean(I**2 + Q**2))
        stats_by_class[label]['phase_std'].append(np.std(np.diff(np.unwrap(phase))))
        
        if np.std(I) > 1e-10 and np.std(Q) > 1e-10:
            stats_by_class[label]['iq_correlation'].append(np.corrcoef(I, Q)[0, 1])
        else:
            stats_by_class[label]['iq_correlation'].append(0)
        
        stats_by_class[label]['snr'].append(snr.item())
    
    # Afficher les statistiques
    print("\n" + "="*80)
    print("SIGNAL STATISTICS BY CLASS")
    print("="*80)
    
    classes = sorted(stats_by_class.keys())
    
    print(f"{'Class':<8} {'Amp Mean':>12} {'Amp Std':>12} {'Power':>12} {'Phase Std':>12} {'IQ Corr':>12}")
    print("-"*80)
    
    for c in classes:
        s = stats_by_class[c]
        name = f"C{c}" if c < 6 else f"Unk{c-6}"
        print(f"{name:<8} {np.mean(s['amplitude_mean']):>12.4f} {np.mean(s['amplitude_std']):>12.4f} "
              f"{np.mean(s['power']):>12.4f} {np.mean(s['phase_std']):>12.4f} {np.mean(s['iq_correlation']):>12.4f}")
    
    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = ['amplitude_mean', 'amplitude_std', 'power', 'phase_std', 'iq_correlation']
    titles = ['Amplitude Mean', 'Amplitude Std', 'Power', 'Phase Variation', 'I-Q Correlation']
    
    for ax, metric, title in zip(axes.flatten()[:5], metrics, titles):
        data = [stats_by_class[c][metric] for c in classes]
        labels_plot = [f"C{c}" if c < 6 else f"Unk" for c in classes]
        colors = ['blue' if c < 6 else 'red' for c in classes]
        
        bp = ax.boxplot(data, labels=labels_plot, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color if color == 'red' else 'lightblue')
            patch.set_alpha(0.6)
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'signal_statistics.png'), dpi=300)
    print(f"\nSaved: {os.path.join(save_dir, 'signal_statistics.png')}")
    plt.close()
    
    return stats_by_class


def analyze_frequency_domain(test_path, save_dir):
    """Analyse le spectre fréquentiel par classe"""
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = SimpleDataset(test_path)
    
    # Calculer le spectre moyen par classe
    spectra_by_class = {}
    
    for idx in range(min(len(dataset), 5000)):  # Limiter pour vitesse
        signal, label, _ = dataset[idx]
        label = label.item()
        
        I, Q = signal[0].numpy(), signal[1].numpy()
        complex_sig = I + 1j * Q
        
        spectrum = np.abs(np.fft.fft(complex_sig))
        spectrum = spectrum[:len(spectrum)//2]  # Moitié positive
        
        if label not in spectra_by_class:
            spectra_by_class[label] = []
        spectra_by_class[label].append(spectrum)
    
    # Moyenner
    mean_spectra = {c: np.mean(spectra, axis=0) for c, spectra in spectra_by_class.items()}
    
    # Visualisation
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Spectres des classes connues
    ax1 = axes[0]
    for c in range(6):
        if c in mean_spectra:
            ax1.plot(mean_spectra[c], label=f'C{c}', alpha=0.8)
    ax1.set_title('Mean Frequency Spectrum - Known Classes (0-5)')
    ax1.set_xlabel('Frequency Bin')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Comparaison connus vs inconnus
    ax2 = axes[1]
    # Classes qui ressemblent aux inconnus
    if 3 in mean_spectra:
        ax2.plot(mean_spectra[3], label='C3 (Known)', color='blue', linewidth=2)
    if 5 in mean_spectra:
        ax2.plot(mean_spectra[5], label='C5 (Known)', color='green', linewidth=2)
    
    # Inconnus
    for c in [6, 7, 8]:
        if c in mean_spectra:
            ax2.plot(mean_spectra[c], label=f'Unk{c-6}', linestyle='--', color='red', alpha=0.7)
    
    ax2.set_title('Spectrum Comparison: C3, C5 vs Unknown Classes')
    ax2.set_xlabel('Frequency Bin')
    ax2.set_ylabel('Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'frequency_analysis.png'), dpi=300)
    print(f"Saved: {os.path.join(save_dir, 'frequency_analysis.png')}")
    plt.close()
    
    return mean_spectra


def visualize_embeddings(classifier_path, test_path, save_dir, device='cuda'):
    """Visualise les embeddings avec t-SNE"""
    from models import CNN_LSTM_SNR_Model
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Charger modèle
    model = CNN_LSTM_SNR_Model(n_classes=6, n_channels=2, hidden_size=64, include_snr=True)
    checkpoint = torch.load(classifier_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    dataset = SimpleDataset(test_path)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    # Extraire features
    features = []
    labels = []
    
    hook_features = []
    def hook(module, input, output):
        hook_features.append(input[0].detach())
    
    handle = model.fc2.register_forward_hook(hook)
    
    with torch.no_grad():
        for signals, lbls, snr in loader:
            signals = signals.to(device)
            snr = snr.to(device).unsqueeze(1)
            _ = model(signals, snr)
            
            features.append(hook_features[-1].cpu().numpy())
            labels.append(lbls.numpy())
    
    handle.remove()
    
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    
    print(f"Features shape: {features.shape}")
    
    # Sous-échantillonner pour t-SNE
    n_samples = min(5000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    features_sub = features[indices]
    labels_sub = labels[indices]
    
    # t-SNE
    print("Computing t-SNE...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=1)
        embeddings_2d = tsne.fit_transform(features_sub)
    except Exception as e:
        print(f"t-SNE failed: {e}")
        print("Falling back to PCA for 2D visualization...")
        pca_2d = PCA(n_components=2)
        embeddings_2d = pca_2d.fit_transform(features_sub)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Toutes les classes
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 9))
    
    for c in range(9):
        mask = labels_sub == c
        if mask.sum() > 0:
            name = f'C{c}' if c < 6 else f'Unk{c-6}'
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[c]], label=name, alpha=0.6, s=20)
    
    ax1.set_title('t-SNE: All Classes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Connus vs Inconnus
    ax2 = axes[1]
    known_mask = labels_sub < 6
    unknown_mask = labels_sub >= 6
    
    ax2.scatter(embeddings_2d[known_mask, 0], embeddings_2d[known_mask, 1], 
               c='blue', label='Known (0-5)', alpha=0.5, s=20)
    ax2.scatter(embeddings_2d[unknown_mask, 0], embeddings_2d[unknown_mask, 1], 
               c='red', label='Unknown (6-8)', alpha=0.7, s=30)
    
    ax2.set_title('t-SNE: Known vs Unknown')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_embeddings.png'), dpi=300)
    print(f"Saved: {os.path.join(save_dir, 'tsne_embeddings.png')}")
    plt.close()
    
    # PCA aussi
    print("Computing PCA...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(features_sub)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for c in range(9):
        mask = labels_sub == c
        if mask.sum() > 0:
            name = f'C{c}' if c < 6 else f'Unk{c-6}'
            color = 'blue' if c < 6 else 'red'
            marker = 'o' if c < 6 else 'x'
            ax.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                      c=color, label=name, alpha=0.5, s=20, marker=marker)
    
    ax.set_title(f'PCA (explained variance: {pca.explained_variance_ratio_.sum():.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_embeddings.png'), dpi=300)
    print(f"Saved: {os.path.join(save_dir, 'pca_embeddings.png')}")
    plt.close()
    
    return features, labels


def analyze_class_confusion(classifier_path, test_path, save_dir, device='cuda'):
    """Analyse détaillée de la confusion entre classes"""
    from models import CNN_LSTM_SNR_Model
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = CNN_LSTM_SNR_Model(n_classes=6, n_channels=2, hidden_size=64, include_snr=True)
    checkpoint = torch.load(classifier_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    dataset = SimpleDataset(test_path)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels, snr in loader:
            signals = signals.to(device)
            snr = snr.to(device).unsqueeze(1)
            logits = model(signals, snr)
            probs = F.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    
    # Analyse pour chaque classe inconnue
    print("\n" + "="*80)
    print("SOFTMAX PROBABILITY ANALYSIS FOR UNKNOWN CLASSES")
    print("="*80)
    
    for unk_class in [6, 7, 8]:
        mask = labels == unk_class
        if mask.sum() == 0:
            continue
        
        unk_probs = probs[mask]
        mean_probs = np.mean(unk_probs, axis=0)
        
        print(f"\nUnknown Class {unk_class-6} (label={unk_class}):")
        print(f"  Samples: {mask.sum()}")
        print(f"  Mean probabilities for each known class:")
        for i, p in enumerate(mean_probs):
            print(f"    C{i}: {p:.4f}")
        print(f"  Most likely prediction: C{np.argmax(mean_probs)} ({np.max(mean_probs):.4f})")
    
    # Visualisation: distribution des probabilités pour les inconnus
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, unk_class in enumerate([6, 7, 8]):
        ax = axes[i]
        mask = labels == unk_class
        if mask.sum() == 0:
            continue
        
        unk_probs = probs[mask]
        
        # Box plot des probabilités pour chaque classe
        ax.boxplot([unk_probs[:, c] for c in range(6)], labels=[f'C{c}' for c in range(6)])
        ax.set_title(f'Unknown {unk_class-6}: Probability Distribution')
        ax.set_ylabel('Softmax Probability')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'unknown_probabilities.png'), dpi=300)
    print(f"\nSaved: {os.path.join(save_dir, 'unknown_probabilities.png')}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default=None)
    parser.add_argument("--test", type=str, default="Data/test_with_unknown.hdf5")
    parser.add_argument("--save_dir", type=str, default="analysis_results")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("SIGNAL STATISTICS ANALYSIS")
    print("="*60)
    analyze_signal_statistics(args.test, args.save_dir)
    
    print("\n" + "="*60)
    print("FREQUENCY DOMAIN ANALYSIS")
    print("="*60)
    analyze_frequency_domain(args.test, args.save_dir)
    
    if args.classifier:
        print("\n" + "="*60)
        print("EMBEDDING VISUALIZATION")
        print("="*60)
        visualize_embeddings(args.classifier, args.test, args.save_dir, device)
        
        print("\n" + "="*60)
        print("CLASS CONFUSION ANALYSIS")
        print("="*60)
        analyze_class_confusion(args.classifier, args.test, args.save_dir, device)


if __name__ == "__main__":
    main()
