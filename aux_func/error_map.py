# evaluation/error_map.py

import numpy as np

def generate_error_map(prediction, ground_truth):
    prediction = (prediction > 0).astype(int)
    ground_truth = (ground_truth > 0).astype(int)
    assert prediction.shape == ground_truth.shape

    error_map = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)

    tp = (prediction == 1) & (ground_truth == 1)
    tn = (prediction == 0) & (ground_truth == 0)
    fp = (prediction == 1) & (ground_truth == 0)
    fn = (prediction == 0) & (ground_truth == 1)

    error_map[tp] = [255, 255, 255]   # 白
    error_map[tn] = [0, 0, 0]         # 黑
    error_map[fp] = [255, 0, 0]       # 红
    error_map[fn] = [0, 255, 0]       # 绿

    return error_map
