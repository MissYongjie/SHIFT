# configs/config.py

import torch

class Config:
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练超参数
    lr_main = 1e-3
    lr_sigma = 1e-2
    num_epochs = 400
    update_interval = 100

    # Scribble标注参数
    target_scribble_ratio = 0.3
    max_increase_per_update = 0.05

    # 特征差异图生成
    margin = 1.0
    weight_consistency = 1.0
    weight_separation = 1.0
    weight_ssim_recon = 1.0
    weight_ssim_min = 1.0

    # 后处理参数
    min_area = 50
    max_components = 2

    # 路径配置
    data_path = '/media/datapart/yongjiezheng/MCD/data/hetdata/dataset#2.mat'
    scribble_path = '/media/datapart/yongjiezheng/MCD/data/scribblefeatures/dataset#2_scribbles.mat'
    result_dir = './results'
    log_dir = './logs'
