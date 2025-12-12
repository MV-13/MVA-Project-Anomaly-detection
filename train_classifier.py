import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from dataset import SignalsDataset
from models import DumpSignalModel, CNN_LSTM_SNR_Model, STFT_CNN_LSTM_SNR_Model


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved at epoch {epoch+1} → {path}")


def validate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    n = 0

    snr_correct = {}
    snr_total = {}

    with torch.no_grad():
        for signals, labels, snr in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)
            snr = snr.to(device).unsqueeze(1)  # shape [B,1]

            outputs = model(signals, snr)
            loss = loss_fn(outputs, labels.long())
            total_loss += loss.item() * signals.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            n += labels.size(0)

            # Track per-SNR accuracy
            for s, p, l in zip(snr, preds, labels):
                s = int(s.item())
                if s not in snr_correct:
                    snr_correct[s] = 0
                    snr_total[s] = 0
                snr_correct[s] += (p == l).item()
                snr_total[s] += 1

    avg_loss = total_loss / n
    accuracy = correct / n

    # Compute per-SNR accuracy
    snr_acc = {s: snr_correct[s] / snr_total[s] for s in snr_correct}

    model.train()
    return avg_loss, accuracy, snr_acc



def main():
    batch_size = 512
    n_epochs = 500
    train_paths = ["Data/train.hdf5", "Data/samples.hdf5"]
    val_path = "Data/validation.hdf5"
    
    # STFT parameters
    transform = None  # None or "stft"
    window_size = 256
    magnitude_only: bool = True 
    
    # Data filtering options
    exclude_zero_snr: bool = False
    only_one_snr: int = -1 # -1 to include all, or snr in {0, 10, 20, 30}
    include_snr: bool = True  # Whether to provide SNR as input to the model
    augment: bool = True  # Whether to apply data augmentation

    # ----------------------------
    # Parse CLI arguments
    # ----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default=None, help="Optional name added to the run folder"
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Path to a checkpoint to load (absolute or relative)"
    )
    args = parser.parse_args()

    # ----------------------------
    # Build run name
    # ----------------------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if args.name is None:
        final_run_name = f"{timestamp}_run"
    else:
        final_run_name = f"{timestamp}_{args.name}"

    log_dir = os.path.join("runs", final_run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    ckpt_path = os.path.join(log_dir, "checkpoint.pt")
    print(f"Logging to {log_dir}")

    # Load datasets
    train_dataset = SignalsDataset(train_paths, transform, magnitude_only=magnitude_only, window_size=window_size, exclude_zero_snr=exclude_zero_snr, only_one_snr=only_one_snr, augment=augment)
    val_dataset = SignalsDataset(val_path, transform, magnitude_only=magnitude_only, window_size=window_size, exclude_zero_snr=exclude_zero_snr, only_one_snr=only_one_snr, augment=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build model
    model = CNN_LSTM_SNR_Model(n_classes=6, n_channels=2, hidden_size=64, include_snr=include_snr)
    print(f"Model has {sum([p.numel() for p in model.parameters()]):,} parameters")
    model.train()

    # ----------------------------
    # Load checkpoint if provided
    # ----------------------------
    if args.load is not None:
        ckpt_path_load = os.path.abspath(args.load)
        print(f"Loading checkpoint from: {ckpt_path_load}")

        if not os.path.isfile(ckpt_path_load):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path_load}")

        checkpoint = torch.load(ckpt_path_load, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model weights from epoch {checkpoint['epoch']+1}")
    else:
        checkpoint = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # If we loaded a checkpoint, restore optimizer state
    if checkpoint is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("Loaded optimizer state")

    # ----------------------------
    # Training Loop
    # ----------------------------
    for epoch in range(n_epochs):
        step = 0
        for signals, labels, snr in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            snr = snr.to(device).unsqueeze(1)  # shape [B,1]

            outputs = model(signals, snr)
            loss_value = loss_fn(outputs, labels.long())

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            writer.add_scalar(
                "Loss/train_step", loss_value.item(), step + epoch * len(train_loader)
            )
            step += 1

        # Compute validation metrics
        val_loss, val_acc, snr_acc = validate(model, val_loader, device, loss_fn)

        # Logging
        writer.add_scalar("Loss/train", loss_value.item(), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Log per-SNR accuracy
        for s, acc in snr_acc.items():
            writer.add_scalar(f"Accuracy/val_SNR_{s}dB", acc, epoch)

        if (epoch + 1) % 10 == 0:
            snr_acc_str = " | ".join([f"SNR={s}dB:{acc:.3f}" for s, acc in snr_acc.items()])
            print(
                f"[{epoch+1}/{n_epochs}] "
                f"train_loss={loss_value.item():.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {snr_acc_str}"
            )

        # -------------------------
        # Save checkpoint every 50 epochs
        # -------------------------
        if (epoch + 1) % 50 == 0:
            ckpt_path_epoch = os.path.join(log_dir, f"checkpoint_epoch{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, loss_value.item(), ckpt_path_epoch)
            print(f"Saved checkpoint at epoch {epoch+1}: {ckpt_path_epoch}")

    # Save final checkpoint
    save_checkpoint(model, optimizer, n_epochs - 1, loss_value.item(), path=ckpt_path)


if __name__ == "__main__":
    main()