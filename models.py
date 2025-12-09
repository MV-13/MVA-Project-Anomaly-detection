import torch
import torch.nn as nn
import torch.nn.functional as F

class DumpSignalModel(nn.Module):
    def __init__(self, n_classes=6, n_channels=2):
        """
        Progressive time-reducing 1D CNN.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=2)
        self.conv5 = nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(1)  # now T=1
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B,16,T/2]
        x = F.relu(self.conv2(x))  # [B,32,T/4]
        x = F.relu(self.conv3(x))  # [B,64,T/8]
        x = F.relu(self.conv4(x))  # [B,32,T/16]
        x = F.relu(self.conv5(x))  # [B,16,T/32]
        x = self.pool(x)  # [B,16,1]
        x = x.view(x.size(0), -1)  # [B,16]
        x = self.fc(x)  # [B,n_classes]
        x = F.softmax(x, dim=1)
        return x


class STFT_CNN_LSTM_SNR_Model(nn.Module):
    def __init__(self, n_classes=6, n_channels=2, hidden_size=64, snr_emb_dim=16, include_snr=True):
        super().__init__()
        self.include_snr = include_snr
        assert n_channels == 2, "Designed for 2-channel STFT input"

        # CNN layers on STFT [B, 2, F, T]
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,2), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Pool only along frequency axis (preserve temporal info)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, T]

        # LSTM on temporal dimension
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)

        # Optional SNR embedding
        if include_snr:
            self.snr_emb = nn.Linear(1, snr_emb_dim)
            self.fc1 = nn.Linear(hidden_size*2 + snr_emb_dim, 128)
        else:
            self.fc1 = nn.Linear(hidden_size*2, 128)

        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x, snr=None):
        # x: [B, 2, F, T] STFT input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Pool frequency axis only
        x = self.freq_pool(x)  # [B, C, 1, T]
        x = x.squeeze(2)       # [B, C, T]
        x = x.permute(0, 2, 1) # [B, T, C] for LSTM

        x, _ = self.lstm(x)
        x = x.mean(dim=1)      # [B, hidden*2]

        if self.include_snr:
            snr = snr.float().unsqueeze(1) if len(snr.shape) == 1 else snr
            snr_feat = F.relu(self.snr_emb(snr))
            x = torch.cat([x, snr_feat], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class CNN_LSTM_SNR_Model(nn.Module):
    def __init__(self, n_classes=6, n_channels=2, hidden_size=64, snr_emb_dim=16, include_snr=True):
        super().__init__()
        self.include_snr = include_snr
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        if include_snr:
            self.snr_emb = nn.Linear(1, snr_emb_dim)  # embed scalar SNR
            self.fc1 = nn.Linear(hidden_size*2 + snr_emb_dim, 128)
        else:
            self.fc1 = nn.Linear(hidden_size*2, 128)

        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x, snr):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # [B, T, C] for LSTM
        x, _ = self.lstm(x)
        x = x.mean(dim=1)       # [B, hidden*2]
        if not self.include_snr:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        snr = snr.float()       # ensure float
        if len(snr.shape) == 1:
            snr = snr.unsqueeze(1)
        snr_feat = F.relu(self.snr_emb(snr))  # [B, snr_emb_dim]

        x = torch.cat([x, snr_feat], dim=1)  # [B, hidden*2 + snr_emb_dim]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Gestion du changement de dimension pour le skip connection (si stride > 1 ou changement de channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out




class ResNetSignalModel(nn.Module):
    def __init__(self, n_classes=6, n_channels=2, base_filters=32, include_snr=True):
        """
        ResNet 1D pour la classification de signaux IQ.
        
        Args:
            n_classes (int): Nombre de classes (6).
            n_channels (int): Canaux d'entrée (2 pour I/Q).
            base_filters (int): Nombre de filtres de départ (contrôle la complexité).
            include_snr (bool): Si True, utilise le SNR comme feature additionnelle.
        """
        super().__init__()
        self.include_snr = include_snr
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(n_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # On double le nombre de filtres périodiquement
        self.layer1 = self._make_layer(base_filters, base_filters, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, blocks=2, stride=2)
        
        # Global Average Pooling: (Batch, Channels, Time) -> (Batch, Channels)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Partie Classifier (Dense)
        dense_input_size = base_filters * 8
        if self.include_snr:
            dense_input_size += 1
            
        self.fc = nn.Sequential(
            nn.Linear(dense_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(128, n_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, snr=None):
        # x shape: [Batch, 2, 2048]
        
        x = self.initial_conv(x)  
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)         # [Batch, 256, 1]
        x = torch.flatten(x, 1)     # [Batch, 256]
        
        if self.include_snr and snr is not None:
            # snr est [Batch, 1]
            norm_snr = snr / 30.0 
            x = torch.cat((x, norm_snr), dim=1) 
            
        x = self.fc(x)
        
        return x