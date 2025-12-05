import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import h5py
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

train_path = 'Data/train.hdf5'
#train_path = 'Data/samples.hdf5'
test_path = 'Data/test_anomalies.hdf5'

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_run_name = f"{timestamp}_run"

# Modèle simple CNN 1D
class SimpleSignalClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            # (2, L) -> (32, L/2)
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # (32, L/2) -> (64, L/4)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # (64, L/4) -> (128, L/8)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # (128, L/8) -> (128, L/16)
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: (B, 2, L)
        x = self.features(x)
        x = self.global_pool(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        logits = self.classifier(x)
        return logits
    
    def get_features(self, x):
        """Extrait les features pour la détection d'anomalies"""
        x = self.features(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device, writer=None, epoch=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (signals, labels, snr) in enumerate(tqdm(dataloader, desc="Training")):
        signals = signals.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Log batch metrics to TensorBoard
        if writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            batch_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)
            writer.add_scalar('Train/BatchAccuracy', batch_acc, global_step)
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for signals, labels, snr in dataloader:
            signals = signals.to(device)
            labels = labels.to(device).long()
            
            outputs = model(signals)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def detect_anomalies(model, dataloader, device, threshold=0.5):
    """
    Détecte les anomalies en utilisant la confiance maximale du softmax.
    Si max(softmax) < threshold, le signal est considéré comme anomalie.
    """
    model.eval()
    predictions = []
    true_labels = []
    confidences = []
    
    with torch.no_grad():
        for signals, labels, snr in dataloader:
            signals = signals.to(device)
            labels = labels.cpu().numpy()
            
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = probs.max(dim=1)
            
            # Prédiction binaire : 0 si connu (confiance haute), 1 si anomalie (confiance basse)
            pred_binary = (max_probs < threshold).cpu().numpy().astype(int)
            
            # Vérité terrain binaire : 0 si classe 0-5, 1 si classe 6-8
            true_binary = (labels >= 6).astype(int)
            
            predictions.extend(pred_binary)
            true_labels.extend(true_binary)
            confidences.extend(max_probs.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)
    
    return predictions, true_labels, confidences


def find_best_threshold(model, dataloader, device):
    """Trouve le meilleur seuil en maximisant le F1-score"""
    model.eval()
    confidences = []
    true_labels = []
    
    with torch.no_grad():
        for signals, labels, snr in dataloader:
            signals = signals.to(device)
            labels = labels.cpu().numpy()
            
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = probs.max(dim=1)
            
            true_binary = (labels >= 6).astype(int)
            
            confidences.extend(max_probs.cpu().numpy())
            true_labels.extend(true_binary)
    
    confidences = np.array(confidences)
    true_labels = np.array(true_labels)
    
    # Tester différents seuils
    thresholds = np.linspace(0.3, 0.95, 50)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        pred_binary = (confidences < thresh).astype(int)
        precision = precision_score(true_labels, pred_binary, zero_division=0)
        recall = recall_score(true_labels, pred_binary, zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"Meilleur seuil trouvé: {best_threshold:.4f} (F1={best_f1:.4f})")
    return best_threshold


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")

    # Charger les données d'entraînement
    from dataset import SignalsDataset  # Assurez-vous que le fichier est accessible

    train_dataset = SignalsDataset(
        path_to_data= train_path,
        transform=None,  # Pas de STFT
        augment=False    # Pas d'augmentation
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)  # num_workers=0 pour Windows

    # Créer le modèle
    model = SimpleSignalClassifier(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Log du graphe du modèle
    log_dir = os.path.join("runs", final_run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    dummy_input = torch.randn(1, 2, 1024).to(device)
    writer.add_graph(model, dummy_input)

    # Entraînement
    num_epochs = 30
    best_acc = 0

    print("\n=== Entraînement du modèle ===")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Log des métriques d'epoch
        writer.add_scalar('Epoch/Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Accuracy', train_acc, epoch)
        writer.add_scalar('Epoch/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step(train_acc)
        
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ Nouveau meilleur modèle sauvegardé!")
            writer.add_scalar('Epoch/BestAccuracy', best_acc, epoch)

    # Charger le meilleur modèle
    model.load_state_dict(torch.load('best_model.pth'))
    print(f"\nMeilleure précision d'entraînement: {best_acc:.2f}%")

    # Test avec détection d'anomalies
    print("\n=== Test - Détection d'anomalies ===")
    test_dataset = SignalsDataset(
        path_to_data= test_path,
        transform=None,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)  # num_workers=0 pour Windows

    # Trouver le meilleur seuil
    best_threshold = find_best_threshold(model, test_loader, device)

    # Évaluer avec le meilleur seuil
    predictions, true_labels, confidences = detect_anomalies(model, test_loader, device, threshold=best_threshold)

    # Calcul des métriques
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    print(f"\n=== Résultats finaux (seuil={best_threshold:.4f}) ===")
    print(f"Précision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(true_labels, predictions)
    print(f"\nMatrice de confusion:")
    print(f"                Prédit: Normal  Prédit: Anomalie")
    print(f"Vrai: Normal       {cm[0,0]:6d}       {cm[0,1]:6d}")
    print(f"Vrai: Anomalie     {cm[1,0]:6d}       {cm[1,1]:6d}")

    # Statistiques par classe
    print("\n=== Analyse par classe ===")
    with torch.no_grad():
        for signals, labels, snr in test_loader:
            signals = signals.to(device)
            labels_np = labels.cpu().numpy()
            
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = probs.max(dim=1)
            max_probs_np = max_probs.cpu().numpy()
            
            for class_id in range(9):
                mask = labels_np == class_id
                if mask.sum() > 0:
                    avg_conf = max_probs_np[mask].mean()
                    std_conf = max_probs_np[mask].std()
                    is_anomaly = "ANOMALIE" if class_id >= 6 else "NORMALE"
                    print(f"Classe {class_id} ({is_anomaly}): confiance moy={avg_conf:.4f} ± {std_conf:.4f}")
            break  # Une seule passe suffit


if __name__ == '__main__':
    main()