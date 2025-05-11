# losses/losses.py

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as pytorch_ssim

def compute_loss(f1, f2, recon_x1, recon_x2, x1, x2, attention_map, 
                 margin=1.0, weight_consistency=1.0, weight_separation=1.0, 
                 weight_ssim_recon=1.0, weight_ssim_min=1.0):
    eps = 1e-6
    device = x1.device
    attention_map = attention_map.to(device)

    # 特征距离
    distance = torch.norm(f1 - f2, p=2, dim=1)

    # 特征一致性损失（未变化区域）
    consistency_weight = (1 - attention_map).squeeze(1)
    consistency_loss = (consistency_weight * distance).sum() / (consistency_weight.sum() + eps)

    # 特征分离损失（变化区域）
    separation_weight = attention_map.squeeze(1)
    margin_diff = torch.clamp(margin - distance, min=0)
    separation_loss = (separation_weight * margin_diff**2).sum() / (separation_weight.sum() + eps)

    # SSIM 损失
    ssim_recon_x1 = pytorch_ssim(recon_x1, x1, data_range=1, size_average=True)
    ssim_recon_x2 = pytorch_ssim(recon_x2, x2, data_range=1, size_average=True)
    ssim_recon_x1_x2 = pytorch_ssim(recon_x1, x2, data_range=1, size_average=True)
    ssim_recon_x2_x1 = pytorch_ssim(recon_x2, x1, data_range=1, size_average=True)

    loss_ssim = (weight_ssim_recon * (2 - ssim_recon_x1 - ssim_recon_x2) +
                 weight_ssim_min * (ssim_recon_x1_x2 + ssim_recon_x2_x1))

    # 总损失
    total_loss = (weight_consistency * consistency_loss + 
                  weight_separation * separation_loss + 
                  loss_ssim)

    return total_loss, consistency_loss, separation_loss, ssim_recon_x1, ssim_recon_x2, ssim_recon_x1_x2, ssim_recon_x2_x1
