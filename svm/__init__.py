"""
Gaussian SVM from Scratch
纯Python实现的高斯核支持向量机

Author: Linyi
Version: 1.0.0
"""

from .core import GaussianSVM
from .visualizer import plot_decision_boundary, plot_training_history, plot_classification_report
from .datasets import (
    generate_ad_detection_data, 
    generate_toy_data, 
    generate_linearly_separable_data,
    generate_xor_data
)

__version__ = "1.0.0"
__author__ = "Linyi"

__all__ = [
    # 核心类
    'GaussianSVM',
    
    # 可视化函数
    'plot_decision_boundary', 
    'plot_training_history',
    'plot_classification_report',
    
    # 数据生成函数
    'generate_ad_detection_data',
    'generate_toy_data',
    'generate_linearly_separable_data',
    'generate_xor_data'
]

# 模块级别的文档
def get_info():
    """获取包信息"""
    return {
        'name': 'Gaussian SVM from Scratch',
        'version': __version__,
        'author': __author__,
        'description': '纯Python实现的高斯核支持向量机',
        'features': [
            'SMO算法实现',
            '高斯核函数',
            '实时训练监控',
            '完整性能评估',
            '丰富的可视化',
            '多种数据生成器'
        ]
    }
