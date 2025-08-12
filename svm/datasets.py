"""
数据集生成模块
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

def generate_ad_detection_data(n_samples: int = 300, 
                              random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成广告检测数据集
    
    特征：[点击率, 停留时间, 关键词密度, 价格敏感度]
    
    Args:
        n_samples: 样本数量
        random_state: 随机种子
        
    Returns:
        X: 特征数据 (n_samples, 4)
        y: 标签数据 (n_samples,) -1表示正常内容，1表示广告
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 正常内容 (非广告, y = -1)
    # 特点：点击率适中，停留时间长，关键词密度低，价格敏感度低
    normal_clickrate = np.random.normal(0.03, 0.01, n_samples//2)  # 3%点击率
    normal_dwelltime = np.random.normal(120, 30, n_samples//2)     # 120秒停留
    normal_keyword_density = np.random.normal(0.02, 0.008, n_samples//2)  # 2%关键词密度
    normal_price_sensitivity = np.random.normal(0.1, 0.05, n_samples//2)  # 低价格敏感度

    # 添加一些非线性相关性：高质量内容停留时间和点击率正相关
    quality_boost = np.random.normal(0, 0.5, n_samples//2)
    normal_clickrate += quality_boost * 0.01
    normal_dwelltime += quality_boost * 20

    X_normal = np.column_stack([normal_clickrate, normal_dwelltime,
                               normal_keyword_density, normal_price_sensitivity])
    y_normal = -np.ones(n_samples//2)

    # 广告内容 (广告, y = 1)
    # 特点：点击率高但停留时间短，关键词密度高，价格敏感度高
    ad_clickrate = np.random.normal(0.08, 0.02, n_samples//2)      # 8%点击率
    ad_dwelltime = np.random.normal(45, 15, n_samples//2)          # 45秒停留
    ad_keyword_density = np.random.normal(0.08, 0.02, n_samples//2)   # 8%关键词密度
    ad_price_sensitivity = np.random.normal(0.6, 0.15, n_samples//2)  # 高价格敏感度

    # 添加广告特有的非线性模式：促销广告关键词密度和价格敏感度强相关
    promo_factor = np.random.normal(0, 1, n_samples//2)
    ad_keyword_density += np.abs(promo_factor) * 0.03
    ad_price_sensitivity += np.abs(promo_factor) * 0.2

    X_ads = np.column_stack([ad_clickrate, ad_dwelltime,
                            ad_keyword_density, ad_price_sensitivity])
    y_ads = np.ones(n_samples//2)

    # 合并数据
    X = np.vstack([X_normal, X_ads])
    y = np.hstack([y_normal, y_ads])

    # 确保所有特征都是正值
    X = np.clip(X, 0.001, None)

    # 标准化特征 (对高斯核很重要)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y

def generate_toy_data(n_samples: int = 200, 
                     noise: float = 0.1,
                     random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成玩具数据集（二维可视化友好）
    
    Args:
        n_samples: 样本数量
        noise: 噪声强度
        random_state: 随机种子
        
    Returns:
        X: 特征数据 (n_samples, 2)
        y: 标签数据 (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 生成两个圆形区域的数据
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    
    # 内圈正类
    theta1 = np.random.uniform(0, 2*np.pi, n_pos)
    r1 = np.random.uniform(0, 2, n_pos)
    X_pos = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    X_pos += np.random.normal(0, noise, X_pos.shape)
    y_pos = np.ones(n_pos)
    
    # 外圈负类
    theta2 = np.random.uniform(0, 2*np.pi, n_neg)
    r2 = np.random.uniform(3, 5, n_neg)
    X_neg = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    X_neg += np.random.normal(0, noise, X_neg.shape)
    y_neg = -np.ones(n_neg)
    
    # 合并数据
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

def generate_linearly_separable_data(n_samples: int = 200,
                                   random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成线性可分数据
    
    Args:
        n_samples: 样本数量  
        random_state: 随机种子
        
    Returns:
        X: 特征数据 (n_samples, 2)
        y: 标签数据 (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 生成两个高斯分布的类别
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    
    # 正类：中心在 (2, 2)
    X_pos = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_pos)
    y_pos = np.ones(n_pos)
    
    # 负类：中心在 (-2, -2)
    X_neg = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], n_neg)
    y_neg = -np.ones(n_neg)
    
    # 合并数据
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

def generate_xor_data(n_samples: int = 200,
                     noise: float = 0.1,
                     random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成XOR模式数据（非线性可分）
    
    Args:
        n_samples: 样本数量
        noise: 噪声强度
        random_state: 随机种子
        
    Returns:
        X: 特征数据 (n_samples, 2)
        y: 标签数据 (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 生成四个区域的数据
    n_per_region = n_samples // 4
    
    # 第一象限和第三象限为正类
    X_pos1 = np.random.multivariate_normal([1, 1], [[0.3, 0], [0, 0.3]], n_per_region)
    X_pos2 = np.random.multivariate_normal([-1, -1], [[0.3, 0], [0, 0.3]], n_per_region)
    X_pos = np.vstack([X_pos1, X_pos2])
    y_pos = np.ones(2 * n_per_region)
    
    # 第二象限和第四象限为负类
    X_neg1 = np.random.multivariate_normal([-1, 1], [[0.3, 0], [0, 0.3]], n_per_region)
    X_neg2 = np.random.multivariate_normal([1, -1], [[0.3, 0], [0, 0.3]], n_samples - 3 * n_per_region)
    X_neg = np.vstack([X_neg1, X_neg2])
    y_neg = -np.ones(len(X_neg))
    
    # 合并数据
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    
    # 添加噪声
    X += np.random.normal(0, noise, X.shape)
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y
