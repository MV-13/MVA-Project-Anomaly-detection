import torch
from torch.utils.data import DataLoader
from dataset import SignalsDataset
from models import CNN_LSTM_SNR_Model
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassAccuracy


def predict_test_set(checkpoint_path, test_path, batch_size=512, magnitude_only=True, 
                     window_size=256, include_snr=True, transform=None):
    """
    Charge un modèle depuis un checkpoint et fait des prédictions sur le test set.
    
    Args:
        checkpoint_path: Chemin vers le fichier checkpoint.pt
        test_path: Chemin vers le fichier test.hdf5
        batch_size: Taille des batchs pour l'inférence
        magnitude_only: Si True, utilise seulement la magnitude pour STFT
        window_size: Taille de la fenêtre pour STFT
        include_snr: Si True, le modèle utilise le SNR comme input
        transform: Type de transformation ("stft" ou None)
    
    Returns:
        predictions: numpy array des prédictions (classes)
        probabilities: numpy array des probabilités pour chaque classe
        true_labels: numpy array des vraies labels
        snr_values: numpy array des valeurs SNR
    """
    
    # Vérifier que le checkpoint existe
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading test dataset from {test_path}...")
    test_dataset = SignalsDataset(
        test_path, 
        transform=transform,
        magnitude_only=magnitude_only,
        window_size=window_size,
        exclude_zero_snr=False,
        only_one_snr=-1,
        augment=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    model = CNN_LSTM_SNR_Model(
        n_classes=6, 
        n_channels=2, 
        hidden_size=64, 
        include_snr=include_snr
    )
    
    # Charger les poids
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from epoch {checkpoint['epoch']+1}")
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_snr = []
    

    print("Making predictions...")
    with torch.no_grad():
        for signals, labels, snr in test_loader:
            signals = signals.to(device)
            snr_input = snr.to(device).unsqueeze(1)  # shape [B,1]
            
            outputs = model(signals, snr_input)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())
            all_snr.append(snr.numpy())
    
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    true_labels = np.concatenate(all_labels)
    snr_values = np.concatenate(all_snr)
    
    return predictions, probabilities, true_labels, snr_values


def compute_metrics(predictions, true_labels, snr_values, n_classes=6):
    """
    Calcule les métriques de performance avec TorchMetrics.
    
    Args:
        predictions: numpy array des prédictions
        true_labels: numpy array des vraies labels
        snr_values: numpy array des valeurs SNR
        n_classes: nombre de classes
    
    Returns:
        dict contenant les métriques
    """
    preds_tensor = torch.from_numpy(predictions)
    labels_tensor = torch.from_numpy(true_labels)
    
    accuracy_metric = MulticlassAccuracy(num_classes=n_classes, average='micro')
    
    # Calculer l'accuracy globale
    overall_accuracy = accuracy_metric(preds_tensor, labels_tensor)
    print(f"\n{'='*50}")
    print(f"OVERALL TEST ACCURACY (TorchMetrics): {overall_accuracy:.4f}")
    print(f"{'='*50}")
    
    # Calculer l'accuracy par SNR
    unique_snr = np.unique(snr_values)
    snr_accuracies = {}
    
    print("\nAccuracy per SNR:")
    for s in sorted(unique_snr):
        mask = snr_values == s
        preds_snr = preds_tensor[mask]
        labels_snr = labels_tensor[mask]
        
        acc_snr = accuracy_metric(preds_snr, labels_snr)
        snr_accuracies[int(s)] = acc_snr.item()
        print(f"  SNR = {int(s):2d} dB: {acc_snr:.4f} ({mask.sum()} samples)")
    
    # Calculer l'accuracy par classe
    print("\nAccuracy per class:")
    class_accuracies = {}
    for c in range(n_classes):
        mask = true_labels == c
        if mask.sum() > 0:
            preds_class = preds_tensor[mask]
            labels_class = labels_tensor[mask]
            acc_class = accuracy_metric(preds_class, labels_class)
            class_accuracies[c] = acc_class.item()
            print(f"  Class {c}: {acc_class:.4f} ({mask.sum()} samples)")
    
    return {
        'overall_accuracy': overall_accuracy.item(),
        'snr_accuracies': snr_accuracies,
        'class_accuracies': class_accuracies
    }


def plot_accuracy_vs_snr(snr_accuracies, save_path=None):
    """
    Crée un graphique de l'accuracy en fonction du SNR.
    
    Args:
        snr_accuracies: dict {snr: accuracy}
        save_path: chemin pour sauvegarder le graphique (optionnel)
    """
    snr_values = sorted(snr_accuracies.keys())
    accuracies = [snr_accuracies[snr] for snr in snr_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Test Accuracy vs SNR', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    for snr, acc in zip(snr_values, accuracies):
        plt.text(snr, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy vs SNR plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrices(predictions, true_labels, snr_values, class_names=None, save_dir=None):
    """
    Crée les matrices de confusion pour chaque SNR.
    
    Args:
        predictions: numpy array des prédictions
        true_labels: numpy array des vraies labels
        snr_values: numpy array des valeurs SNR
        class_names: liste des noms de classes (optionnel)
        save_dir: dossier pour sauvegarder les matrices (optionnel)
    """
    unique_snr = sorted(np.unique(snr_values))
    n_snr = len(unique_snr)
    
    n_cols = min(2, n_snr)
    n_rows = (n_snr + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    
    if n_snr == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    if class_names is None:
        n_classes = len(np.unique(true_labels))
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    for idx, snr in enumerate(unique_snr):
        # Filtrer les données pour ce SNR
        mask = snr_values == snr
        preds_snr = predictions[mask]
        labels_snr = true_labels[mask]
        
        cm = confusion_matrix(labels_snr, preds_snr)
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx], cbar_kws={'label': 'Normalized Count'})
        
        # Calculer l'accuracy pour ce SNR
        accuracy = (preds_snr == labels_snr).mean()
        
        axes[idx].set_title(f'SNR = {int(snr)} dB (Acc: {accuracy:.3f})', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Class', fontsize=10)
        axes[idx].set_ylabel('True Class', fontsize=10)
    
    for idx in range(n_snr, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    
    plt.show()
    
    # Matrice de confusion globale
    plt.figure(figsize=(10, 8))
    cm_overall = confusion_matrix(true_labels, predictions)
    cm_overall_normalized = cm_overall.astype('float') / cm_overall.sum(axis=1, keepdims=True)
    
    sns.heatmap(cm_overall_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    overall_accuracy = (predictions == true_labels).mean()
    plt.title(f'Overall Confusion Matrix (Acc: {overall_accuracy:.3f})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'confusion_matrix_overall.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Overall confusion matrix saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict on test set and compute metrics")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to checkpoint file (e.g., runs/20241121-123456_run/checkpoint.pt)"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        default="test.hdf5",
        help="Path to test dataset (default: test.hdf5)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=512,
        help="Batch size for inference (default: 512)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save results and plots (default: same as checkpoint)"
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs='+',
        default=None,
        help="Names of classes (optional, e.g., --class_names BPSK QPSK 8PSK 16QAM 64QAM GMSK)"
    )
    
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.checkpoint)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prédictions
    print("\n" + "="*50)
    print("STEP 1: Making predictions on test set")
    print("="*50)
    predictions, probabilities, true_labels, snr_values = predict_test_set(
        checkpoint_path=args.checkpoint,
        test_path=args.test,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*50)
    print("STEP 2: Computing metrics with TorchMetrics")
    print("="*50)
    metrics = compute_metrics(predictions, true_labels, snr_values)
    
    # Graphique Accuracy vs SNR
    print("\n" + "="*50)
    print("STEP 3: Plotting Accuracy vs SNR")
    print("="*50)
    plot_accuracy_vs_snr(
        metrics['snr_accuracies'],
        save_path=os.path.join(args.save_dir, 'accuracy_vs_snr.png')
    )
    
    # Matrices de confusion
    print("\n" + "="*50)
    print("STEP 4: Plotting Confusion Matrices")
    print("="*50)
    plot_confusion_matrices(
        predictions, 
        true_labels, 
        snr_values,
        class_names=args.class_names,
        save_dir=args.save_dir
    )
    
    # Sauvegarder les résultats numériques
    results_path = os.path.join(args.save_dir, 'test_results.npz')
    np.savez(
        results_path,
        predictions=predictions,
        probabilities=probabilities,
        true_labels=true_labels,
        snr_values=snr_values,
        overall_accuracy=metrics['overall_accuracy'],
        snr_accuracies=metrics['snr_accuracies'],
        class_accuracies=metrics['class_accuracies']
    )
    print(f"\nAll results saved to {args.save_dir}")
    print("="*50)


if __name__ == "__main__":
    main()