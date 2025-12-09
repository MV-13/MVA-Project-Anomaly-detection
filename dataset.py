import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import math

class SignalsDataset(Dataset):
    def __init__(self, path_to_data, transform=None, magnitude_only=True, window_size=256, augment=True):
        """
        Dataset flexible :
        - Si transform=None : Renvoie (3, L) -> I, Q, Amplitude
        - Si transform="stft" : Renvoie (1, F, T) -> Spectrogramme
        """
        super().__init__()
        self.paths = path_to_data
        self.transform = transform
        self.magnitude_only = magnitude_only
        self.nfft = window_size
        self.augment = augment

        # Chargement (gestion liste ou str)
        if isinstance(path_to_data, str):
            path_to_data = [path_to_data]
            
        self.signals = []
        self.labels = []
        self.snr = []
        
        for path in path_to_data:
            with h5py.File(path, "r") as f:
                self.signals.append(np.array(f["signaux"], dtype=np.float32))
                self.labels.append(np.array(f["labels"], dtype=np.int8))
                self.snr.append(np.array(f["snr"], dtype=np.int16))
                
        self.signals = np.concatenate(self.signals, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.snr = np.concatenate(self.snr, axis=0)

        print(f"Dataset loaded: {len(self.signals)} samples.")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # 1. Récupération Brute (2, L)
        signal = self.signals[idx] # (N, 2, L) -> ici (2, L) stocké souvent comme (N, L, 2) ou (N, 2, L)
        # Vérifions la shape. Le fichier HDF5 donne souvent (N, 2, L) ou (N, L, 2).
        # Dans le TP standard : (N, 2, L) -> signal est (2, L)
        
        signal = torch.tensor(signal, dtype=torch.float32)
        
        # Si shape est (L, 2), on transpose
        if signal.shape[0] != 2 and signal.shape[1] == 2:
            signal = signal.transpose(0, 1) # (2, L)

        label = torch.tensor(self.labels[idx], dtype=torch.int8)
        snr = torch.tensor(self.snr[idx], dtype=torch.float32)

        # 2. Augmentation (Rotation de phase) - Seulement sur I/Q
        if self.augment:
            angle = 2 * math.pi * torch.rand(1).item()
            I, Q = signal[0], signal[1]
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            I_rot = I * cos_a - Q * sin_a
            Q_rot = I * sin_a + Q * cos_a
            signal = torch.stack([I_rot, Q_rot], dim=0)

        # ---------------------------------------------------------
        # CAS 1 : SPECTROGRAMME (STFT)
        # ---------------------------------------------------------
        if self.transform == "stft":
            # On forme un signal complexe pour la STFT : I + jQ
            complex_signal = torch.complex(signal[0], signal[1]) # (L,)
            
            # Calcul STFT
            spec = torch.stft(
                complex_signal,
                n_fft=self.nfft,
                hop_length=self.nfft // 2,
                win_length=self.nfft,
                window=torch.hann_window(self.nfft, device=signal.device),
                return_complex=True,
                center=False # Pour éviter le padding excessif
            ) # Résultat : (Freq, Time) complexe
            
            if self.magnitude_only:
                spec = torch.abs(spec) # (Freq, Time) magnitude
                # Ajout dimension canal pour le CNN 2D : (1, F, T)
                signal = spec.unsqueeze(0)
            else:
                # Si on voulait garder la phase (optionnel)
                signal = torch.view_as_real(spec).permute(2, 0, 1)

        # ---------------------------------------------------------
        # CAS 2 : TEMPOREL (I, Q, Amplitude)
        # ---------------------------------------------------------
        else:
            # Calcul de l'amplitude
            amp = torch.sqrt(signal[0]**2 + signal[1]**2)
            # Stack pour avoir (3, L)
            signal = torch.cat([signal, amp.unsqueeze(0)], dim=0)

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