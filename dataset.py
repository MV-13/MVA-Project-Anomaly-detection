import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import math

class SignalsDataset(Dataset):
    def __init__(self, path_to_data: list[str] | str, transform=None, magnitude_only=True, window_size=256, exclude_zero_snr=False, only_one_snr=-1, augment=True):
        """
        Dataset PyTorch for signalsfrom a HDF5 file.
        we keep labels in int8, snr in int16.
        """
        super().__init__()
        self.paths = path_to_data
        self.transform = transform
        self.magnitude_only = magnitude_only
        self.nfft = window_size
        self.augment = augment

        if isinstance(path_to_data, list):
            self.signals = []
            self.labels = []
            self.snr = []
            for path in path_to_data:
                with h5py.File(path, "r") as f:
                    self.signals.append(np.array(f["signaux"], dtype=np.float32))  # (N, 2, L)
                    self.labels.append(np.array(f["labels"], dtype=np.int8))  # (N,)
                    self.snr.append(np.array(f["snr"], dtype=np.int16))  # (N,)
            self.signals = np.concatenate(self.signals, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            self.snr =  np.concatenate(self.snr, axis=0)
        else:
            with h5py.File(self.paths, "r") as f:
                self.signals = np.array(f["signaux"], dtype=np.float32)  # (N, 2, L)
                self.labels = np.array(f["labels"], dtype=np.int8)  # (N,)
                self.snr = np.array(f["snr"], dtype=np.int16)  # (N,)
        
        if only_one_snr != -1:
            mask = self.snr == only_one_snr
            before = len(self.snr)
            self.signals = self.signals[mask]
            self.labels = self.labels[mask]
            self.snr = self.snr[mask]
            print(f"Filtered dataset to only include SNR = {only_one_snr}, kept {len(self.snr)} samples out of {before}")

        if exclude_zero_snr:
            mask = self.snr != 0
            before = len(self.snr)
            self.signals = self.signals[mask]
            self.labels = self.labels[mask]
            self.snr = self.snr[mask]
            print(f"Removed {before - len(self.snr)} samples with SNR = 0")
        print(
            f"Dataset loaded: {len(self.signals)} samples, each of shape {self.signals[0].shape}"
        )

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]  # (2, L)
        label = self.labels[idx]  # int8
        snr = self.snr[idx]  # int16

        # Convertir en Tensor
        signal = torch.tensor(signal, dtype=torch.float32).transpose(
            -1, -2
        )  # (L, 2) -> (2, L)
        if self.augment:
            signal = random_rotate_iq(signal) # Data augmentation: random IQ rotation
            
            # SNR-based noise augmentation
            if torch.rand(1).item() < 0.5:
                if snr.item() == 30:
                    new_snr = torch.tensor(np.random.choice([20, 10, 0]), dtype=torch.float32)
                    signal = add_noise_to_snr(signal, snr.item(), new_snr.item())
                    snr = new_snr
                elif snr.item() == 20:
                    new_snr = torch.tensor(np.random.choice([10, 0]), dtype=torch.float32)
                    signal = add_noise_to_snr(signal, snr.item(), new_snr.item())
                    snr = new_snr
                elif snr.item() == 10:
                    new_snr = torch.tensor(0, dtype=torch.float32)
                    signal = add_noise_to_snr(signal, snr.item(), new_snr.item())
                    snr = new_snr

        if self.transform == "stft":
            spec = torch.stft(
                signal,
                n_fft=self.nfft,
                hop_length=self.nfft//2,
                win_length=self.nfft,
                window=torch.hann_window(self.nfft, device=signal.device),
                return_complex=True,
            )  # (C, F, T)
            if self.magnitude_only:
                signal = torch.abs(spec)  # (C, F, T)
            else:
                spec = torch.view_as_real(spec)  # (C, F, T, 2)
                spec = spec.permute(0, 2, 1, 3).contiguous()
                signal = spec.view(spec.size(0), spec.size(1), -1)
        label = torch.tensor(label, dtype=torch.int8)  # keep it as int8
        snr = torch.tensor(snr, dtype=torch.float32)  # float32 for model input

        return signal, label, snr


def rotate_iq(signal, angle_rad):
    """
    Rotate IQ signal by a given angle in radians.
    signal: tensor of shape [2, L] (I, Q)
    angle_rad: rotation angle in radians
    """
    I = signal[0]
    Q = signal[1]
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    I_rot = I * cos_theta - Q * sin_theta
    Q_rot = I * sin_theta + Q * cos_theta
    rotated = torch.stack([I_rot, Q_rot], dim=0)
    return rotated

def random_rotate_iq(signal):
    angle = 2 * math.pi * torch.rand(1).item()  # random angle 0 -> 2π
    return rotate_iq(signal, angle)


def add_noise_to_snr(signal, current_snr, target_snr):
    """
    Add Gaussian noise to the signal to simulate a lower SNR.
    signal: tensor of shape [2, L]
    current_snr: float, current SNR in dB
    target_snr: float, target SNR in dB
    """
    if target_snr >= current_snr:
        return signal  # don't increase SNR

    # Calculate linear scale SNR
    current_linear = 10 ** (current_snr / 10)
    target_linear = 10 ** (target_snr / 10)
    
    # Signal power
    power_signal = signal.pow(2).mean()
    
    # Noise power required to reach target SNR
    power_noise = power_signal / target_linear
    
    # Current noise power
    noise_current = power_signal / current_linear
    
    # Additional noise to add
    additional_noise_power = max(0, power_noise - noise_current)
    
    if additional_noise_power > 0:
        noise = torch.randn_like(signal) * torch.sqrt(additional_noise_power)
        return signal + noise
    else:
        return signal