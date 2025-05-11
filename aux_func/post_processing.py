# evaluation/post_processing.py

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

def generate_weighted_diff_map(diff_map, scribble):
    distance_map = distance_transform_edt(1 - scribble)
    distance_map_norm = distance_map / np.max(distance_map)
    in_distance_map_norm = 1 - distance_map_norm
    weighted_diff_map = diff_map * in_distance_map_norm
    return weighted_diff_map, in_distance_map_norm

def adaptive_morphological_filtering(binary_map, in_distance_map_norm, min_area=5, max_components=None):
    close_threshold = 0.9
    medium_threshold = 0.5

    close_mask = in_distance_map_norm >= close_threshold
    medium_mask = (in_distance_map_norm < close_threshold) & (in_distance_map_norm >= medium_threshold)
    far_mask = in_distance_map_norm < medium_threshold

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_far = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    iterations_close = 1
    iterations_medium = 1
    iterations_far = 2

    bcm_close = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel_close, iterations=iterations_close)
    bcm_close = cv2.morphologyEx(bcm_close, cv2.MORPH_CLOSE, kernel_close, iterations=iterations_close)

    bcm_medium = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel_medium, iterations=iterations_medium)
    bcm_medium = cv2.morphologyEx(bcm_medium, cv2.MORPH_CLOSE, kernel_medium, iterations=iterations_medium)

    bcm_far = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel_far, iterations=iterations_far)
    bcm_far = cv2.morphologyEx(bcm_far, cv2.MORPH_CLOSE, kernel_far, iterations=iterations_far)

    processed_map = np.zeros_like(binary_map)
    processed_map[close_mask] = bcm_close[close_mask]
    processed_map[medium_mask] = bcm_medium[medium_mask]
    processed_map[far_mask] = bcm_far[far_mask]

    num_labels, labels_im = cv2.connectedComponents(processed_map)

    component_areas = []
    for label in range(1, num_labels):
        area = np.sum(labels_im == label)
        if area >= min_area:
            component_areas.append((area, label))

    component_areas.sort(reverse=True)
    if max_components is not None:
        component_areas = component_areas[:max_components]

    filtered_map = np.zeros_like(labels_im, dtype=np.uint8)
    for area, label in component_areas:
        filtered_map[labels_im == label] = 255

    return filtered_map
