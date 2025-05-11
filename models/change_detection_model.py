# models/change_detection_model.py

import torch.nn as nn
from models.base_model import EncoderDecoder
from models.gaussian_blur import GaussianBlur

class ChangeDetectionModel(nn.Module):
    def __init__(self, in_channels, feature_dim=64, kernel_size=11, sigma_init=2.0):
        super(ChangeDetectionModel, self).__init__()
        self.model_t1 = EncoderDecoder(in_channels, feature_dim)
        self.model_t2 = EncoderDecoder(in_channels, feature_dim)
        self.gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma_init=sigma_init)

    def forward(self, x1, x2, scribble):
        f1, recon_x1 = self.model_t1(x1)
        f2, recon_x2 = self.model_t2(x2)
        attention_map = self.gaussian_blur(scribble)
        return f1, f2, recon_x1, recon_x2, attention_map
