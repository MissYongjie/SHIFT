import numpy as np


def preprocess_img(data, d_type, norm_type):
    pps_data = np.array(data).astype(np.float32)
    if d_type == 'opt':
        if norm_type == 'stad':
            pps_data = stad_img(pps_data, channel_first=False)
        else:
            pps_data = norm_img(pps_data, channel_first=False)
    elif d_type == 'sar':
        pps_data[np.abs(pps_data) <= 0] = np.min(pps_data[np.abs(pps_data) > 0])
        pps_data = np.log(pps_data + 1.0)
        if norm_type == 'stad':
            pps_data = stad_img(pps_data, channel_first=False)
            # sigma = np.std(pps_data)
            # mean = np.mean(pps_data)
            # idx_min = pps_data < (mean - 4 * sigma)
            # idx_max = pps_data > (mean + 4 * sigma)
            # pps_data[idx_min] = np.min(pps_data[~idx_min])
            # pps_data[idx_max] = np.max(pps_data[~idx_max])
        else:
            pps_data = norm_img(pps_data, channel_first=False)
            # sigma = np.std(pps_data)
            # mean = np.mean(pps_data)
            # idx_min = pps_data < (mean - 4* sigma)
            # idx_max = pps_data > (mean + 4 * sigma)
            # pps_data[idx_min] = np.min(pps_data[~idx_min])
            # pps_data[idx_max] = np.max(pps_data[~idx_max])
    return pps_data


def norm_img(img, channel_first):
    '''
        normalize value to [0, 1]
    '''
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        max_value = np.max(img, axis=1, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=1, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = (img - min_value) / diff_value
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (channel, height * width)
        max_value = np.max(img, axis=0, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=0, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = (img - min_value) / diff_value
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img


def norm_img_2(img, channel_first):
    '''
    normalize value to [-1, 1]
    '''
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        max_value = np.max(img, axis=1, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=1, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = 2 * ((img - min_value) / diff_value - 0.5)
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (channel, height * width)
        max_value = np.max(img, axis=0, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=0, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = 2 * ((img - min_value) / diff_value - 0.5)
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img


def stad_img(img, channel_first):
    """
    normalization image
    :param channel_first:
    :param img: (C, H, W)
    :return:
        norm_img: (C, H, W)
    """
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        mean = np.mean(img, axis=1, keepdims=True)  # (channel, 1)
        center = img - mean  # (channel, height * width)
        var = np.sum(np.power(center, 2), axis=1, keepdims=True) / (img_height * img_width)  # (channel, 1)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (height * width, channel)
        mean = np.mean(img, axis=0, keepdims=True)  # (1, channel)
        center = img - mean  # (height * width, channel)
        var = np.sum(np.power(center, 2), axis=0, keepdims=True) / (img_height * img_width)  # (1, channel)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img

def compute_superpixel_features(img, objects):
    """
    计算每个超像素的均值、最大值和最小值特征。

    Parameters:
    - img: np.array, 输入图像，假设为多通道图像 (H, W, C)。
    - objects: np.array, 超像素标签图 (H, W)，每个像素值代表其所属的超像素 ID。

    Returns:
    - features: dict, 每个超像素的特征，键为超像素 ID，值为包含均值、最大值和最小值的字典。
    """
    # 存储每个超像素的特征
    features = {}

    # 遍历每个超像素区域
    for superpixel_id in np.unique(objects):
        # 获取当前超像素区域的掩码
        mask = (objects == superpixel_id)
        
        # 提取该超像素区域的像素值
        pixels = img[mask]
        
        # 计算均值、最大值和最小值
        mean_value = pixels.mean(axis=0)
        max_value = pixels.max(axis=0)
        min_value = pixels.min(axis=0)
        
        # 存储每个超像素区域的特征
        features[superpixel_id] = {
            "mean": mean_value,
            "max": max_value,
            "min": min_value
        }

    return features

import numpy as np

def compute_superpixel_features_image(img, objects):
    """
    计算每个超像素的均值、最大值和最小值特征，并返回大小为 (H, W, channel) 的特征图。

    Parameters:
    - img: np.array, 输入图像，假设为多通道图像 (H, W, C)。
    - objects: np.array, 超像素标签图 (H, W)，每个像素值代表其所属的超像素 ID。

    Returns:
    - mean_img: np.array, 每个超像素区域的均值图像 (H, W, C)。
    - max_img: np.array, 每个超像素区域的最大值图像 (H, W, C)。
    - min_img: np.array, 每个超像素区域的最小值图像 (H, W, C)。
    """
    # 获取图像的高度、宽度和通道数
    height, width, channels = img.shape
    
    # 初始化存储特征图像的数组
    mean_img = np.zeros_like(img, dtype=np.float32)
    max_img = np.zeros_like(img, dtype=np.float32)
    min_img = np.zeros_like(img, dtype=np.float32)

    # 遍历每个超像素区域
    for superpixel_id in np.unique(objects):
        # 获取当前超像素区域的掩码
        mask = (objects == superpixel_id)
        
        # 提取该超像素区域的像素值
        pixels = img[mask]
        
        # 计算均值、最大值和最小值
        mean_value = pixels.mean(axis=0)
        max_value = pixels.max(axis=0)
        min_value = pixels.min(axis=0)
        
        # 将计算出的特征值填充到整个超像素区域
        mean_img[mask] = mean_value
        max_img[mask] = max_value
        min_img[mask] = min_value

    return mean_img, max_img, min_img

def normalize_per_band(img, method='std'):
    """
    对图像逐个波段进行归一化。

    参数：
    - img: NumPy数组，形状为 (H, W, C)
    - method: 字符串，'std' 表示标准化（均值0，标准差1），'norm' 表示归一化到 [0, 1]

    返回：
    - normalized_img: 归一化后的图像，形状与输入相同
    """
    assert method in ['std', 'norm'], "method 参数必须是 'std' 或 'norm'"

    img = img.astype(np.float32)
    normalized_img = np.zeros_like(img)

    for c in range(img.shape[2]):
        band = img[:, :, c]

        if method == 'std':
            mean = band.mean()
            std = band.std()
            normalized_band = (band - mean) / (std + 1e-6)  # 防止除以零
        elif method == 'norm':
            min_val = band.min()
            max_val = band.max()
            normalized_band = (band - min_val) / (max_val - min_val + 1e-6)  # 防止除以零

        normalized_img[:, :, c] = normalized_band

    return normalized_img
