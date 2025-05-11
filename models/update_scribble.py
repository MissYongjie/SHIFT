# scribble_update/update_scribble.py

import numpy as np
import cv2

def update_scribble_confidence(scribble, distance_map, original_img, current_ratio, 
                                target_ratio=0.3, max_increase_per_update=0.05, 
                                dilation_iterations=5, top_percentage=0.05):
    H, W = scribble.shape

    remaining_ratio = target_ratio - current_ratio
    if remaining_ratio <= 0:
        print("达到目标覆盖率，停止更新。")
        return scribble

    increase_ratio = min(max_increase_per_update, remaining_ratio)
    target_new_ratio = current_ratio + increase_ratio
    target_pixels = int(target_new_ratio * H * W)

    kernel = np.ones((33, 33), np.uint8)
    scribble_dilated = cv2.dilate(scribble, kernel, iterations=dilation_iterations)
    candidate_mask = np.logical_and(scribble_dilated, scribble == 0)
    candidate_distances = distance_map * candidate_mask

    num_candidates = int(np.sum(candidate_mask))
    if num_candidates == 0:
        print("没有候选区域。")
        return scribble

    num_new_pixels = max(1, int(top_percentage * num_candidates))
    candidate_indices = np.argwhere(candidate_distances > 0)
    candidate_values = distance_map[candidate_mask]

    num_new_pixels = min(num_new_pixels, len(candidate_values))
    sorted_indices = np.argsort(candidate_values)[::-1]
    top_pixels = candidate_indices[sorted_indices[:num_new_pixels]]

    new_scribble = scribble.copy()
    for y, x in top_pixels:
        new_scribble[y, x] = 1

    return new_scribble.astype(np.uint8)
