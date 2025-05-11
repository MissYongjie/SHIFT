# models/base_model.py

import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, in_channels, feature_dim=64):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )

    def forward(self, x):
        features = self.encoder(x)
        recon_x = self.decoder(features)
        return features, recon_x
