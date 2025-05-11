# main.py

import os
import numpy as np
import torch
import scipy.io
from torch.utils.data import DataLoader
from imageio import imsave
from configs.config import Config
from datasets.dataset import ChangeDetectionDataset
from models.change_detection_model import ChangeDetectionModel
from losses.losses import compute_loss
from models.update_scribble import update_scribble_confidence
from aux_func.metrics import compute_metrics
from aux_func.error_map import generate_error_map
from aux_func.post_processing import generate_weighted_diff_map, adaptive_morphological_filtering
from skimage.filters import threshold_otsu
from torch.utils.tensorboard import SummaryWriter

def main():
    cfg = Config()

    os.makedirs(cfg.result_dir, exist_ok=True)

    # 1. 加载数据
    data = scipy.io.loadmat(cfg.data_path)
    scribs = scipy.io.loadmat(cfg.scribble_path)
    img_t1, img_t2, ref_gt = data['image_t1'], data['image_t2'], data['Ref_gt']
    # scribble = data['scribble']
    img_t1 = img_t1.astype(np.float32) / 255.0
    img_t2 = img_t2.astype(np.float32) / 255.0
    ground_truth = (ref_gt > 0).astype(np.uint8) * 255

    # choose a specific index for scribble testing
    i = 57
    scribble = scribs['all_data']['scribble'][0][i].squeeze()
    scribble = (scribble > 0).astype(np.float32)

    dataset = ChangeDetectionDataset(img_t1, img_t2, scribble)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ChangeDetectionModel(in_channels=img_t1.shape[2]).to(cfg.device)

    optimizer = torch.optim.Adam([
        {'params': model.model_t1.parameters()},
        {'params': model.model_t2.parameters()},
        {'params': model.gaussian_blur.log_sigma, 'lr': cfg.lr_sigma}
    ], lr=cfg.lr_main)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    writer = SummaryWriter(cfg.log_dir)

    # 2. 训练
    for epoch in range(cfg.num_epochs):
        model.train()
        for img_t1_batch, img_t2_batch, scribble_batch in dataloader:
            img_t1_batch = img_t1_batch.to(cfg.device)
            img_t2_batch = img_t2_batch.to(cfg.device)
            scribble_batch = scribble_batch.to(cfg.device)

            f1, f2, recon_x1, recon_x2, attention_map = model(img_t1_batch, img_t2_batch, scribble_batch)
            loss, *_ = compute_loss(f1, f2, recon_x1, recon_x2, img_t1_batch, img_t2_batch, attention_map,
                                    margin=cfg.margin,
                                    weight_consistency=cfg.weight_consistency,
                                    weight_separation=cfg.weight_separation,
                                    weight_ssim_recon=cfg.weight_ssim_recon,
                                    weight_ssim_min=cfg.weight_ssim_min)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(loss.item())

        # 更新scribble
        if (epoch + 1) % cfg.update_interval == 0:
            model.eval()
            with torch.no_grad():
                distance = torch.norm(f1 - f2, p=2, dim=1).squeeze().cpu().numpy()
            current_coverage = np.sum(scribble) / (img_t1.shape[0] * img_t1.shape[1])
            scribble = update_scribble_confidence(scribble, distance, img_t2, current_coverage,
                                                  target_ratio=cfg.target_scribble_ratio,
                                                  max_increase_per_update=cfg.max_increase_per_update)
            dataset.update_scribble(scribble)

        print(f'Epoch [{epoch+1}/{cfg.num_epochs}], Loss: {loss.item():.4f}')

    writer.close()

    # 3. 评估
    model.eval()
    with torch.no_grad():
        img_t1_tensor = torch.from_numpy(img_t1.transpose(2, 0, 1)).unsqueeze(0).to(cfg.device)
        img_t2_tensor = torch.from_numpy(img_t2.transpose(2, 0, 1)).unsqueeze(0).to(cfg.device)
        scribble_tensor = torch.from_numpy(scribble).unsqueeze(0).unsqueeze(0).float().to(cfg.device)

        f1, f2, *_ = model(img_t1_tensor, img_t2_tensor, scribble_tensor)
        distance = torch.norm(f1 - f2, p=2, dim=1).squeeze().cpu().numpy()

    diff_map = (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
    threshold = threshold_otsu(diff_map.flatten())
    pred_map = (diff_map > threshold).astype(np.uint8) * 255

    oa, f1_score, kappa, conf_mat = compute_metrics(ground_truth, pred_map)
    print(f"OA: {oa:.3f}, F1: {f1_score:.3f}, Kappa: {kappa:.3f}")

    # 保存结果
    imsave(os.path.join(cfg.result_dir, 'pred_map.png'), pred_map)
    imsave(os.path.join(cfg.result_dir, 'error_map.png'), generate_error_map(pred_map, ground_truth))

    # 4. 后处理
    weighted_diff_map, in_distance_map_norm = generate_weighted_diff_map(diff_map, scribble)
    weighted_threshold = threshold_otsu(weighted_diff_map.flatten())
    weighted_pred = (weighted_diff_map > weighted_threshold).astype(np.uint8) * 255
    weighted_pred = adaptive_morphological_filtering(weighted_pred, in_distance_map_norm, 
                                                     min_area=cfg.min_area, 
                                                     max_components=cfg.max_components)

    oa_w, f1_w, kappa_w, _ = compute_metrics(ground_truth, weighted_pred)
    print(f"(Post-processed) OA: {oa_w:.3f}, F1: {f1_w:.3f}, Kappa: {kappa_w:.3f}")

    imsave(os.path.join(cfg.result_dir, 'weighted_pred_map.png'), weighted_pred)

if __name__ == "__main__":
    main()
