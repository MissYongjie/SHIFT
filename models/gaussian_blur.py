# models/gaussian_blur.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=11, sigma_init=2.0):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma_init)))

    def forward(self, scribble):
        sigma = torch.exp(self.log_sigma)
        coord = torch.arange(self.kernel_size, dtype=scribble.dtype, device=scribble.device)
        coord = coord - self.kernel_size // 2
        x_grid, y_grid = torch.meshgrid(coord, coord, indexing='ij')
        gaussian_kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        padding = self.kernel_size // 2
        attention_map = F.conv2d(scribble, gaussian_kernel, padding=padding)
        return attention_map
