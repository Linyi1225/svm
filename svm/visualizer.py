"""
SVM可视化工具模块
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import GaussianSVM

def plot_decision_boundary(svm: 'GaussianSVM', X: np.ndarray, y: np.ndarray, 
                          h: float = 0.01, save_path: Optional[str] = None) -> None:
    """
    绘制SVM决策边界
    
    Args:
        svm: 训练好的GaussianSVM模型
        X: 特征数据，只使用前两个特征进行可视化
        y: 标签数据
        h: 网格步长
        save_path: 保存路径，如果提供则保存图片
    """
    # 只使用前两个特征进行可视化
    X_plot = X[:, :2]
    
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 处理多维特征
    n_features = svm.X.shape[1]
    if n_features > 2:
        grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
        mean_other_features = np.mean(svm.X[:, 2:], axis=0)
        grid_points = np.hstack((grid_points_2d, np.tile(mean_other_features, (grid_points_2d.shape[0], 1))))
    else:
        grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z = svm.decision_function(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 8))

    # 绘制等高线
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    contour_lines = plt.contour(xx, yy, Z, levels=[-1, 0, 1], 
                               colors=['red', 'black', 'blue'], 
                               linestyles=['--', '-', '--'], 
                               linewidths=[1, 2, 1])
    plt.clabel(contour_lines, inline=True, fontsize=8)

    # 绘制数据点
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap='RdYlBu', 
                         s=50, edgecolors='black', alpha=0.7)

    # 高亮支持向量
    if hasattr(svm, 'support_vectors_'):
        support_vectors_plot = svm.support_vectors_[:, :2]
        plt.scatter(support_vectors_plot[:, 0], support_vectors_plot[:, 1],
                   s=120, facecolors='none', edgecolors='yellow', linewidths=3, 
                   label=f'支持向量 ({len(svm.support_vectors_)}个)')

    plt.colorbar(scatter, label='类别')
    plt.title(f'SVM决策边界\nC={svm.C}, γ={svm.gamma}, 训练时间={svm.training_time:.2f}s')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"决策边界图已保存到: {save_path}")
    
    plt.show()

def plot_training_history(svm: 'GaussianSVM', save_path: Optional[str] = None) -> None:
    """
    绘制训练历史
    
    Args:
        svm: 训练好的GaussianSVM模型
        save_path: 保存路径
    """
    if not hasattr(svm, 'training_history') or not svm.training_history:
        print("没有训练历史数据")
        return
    
    history = svm.training_history
    iterations = [h['iteration'] for h in history]
    accuracies = [h['accuracy'] for h in history]
    num_svs = [h['num_sv'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 准确率变化
    ax1.plot(iterations, accuracies, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('准确率')
    ax1.set_title('训练准确率变化')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # 支持向量数量变化
    ax2.plot(iterations, num_svs, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('支持向量数量')
    ax2.set_title('支持向量数量变化')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
    
    plt.show()

def plot_classification_report(svm: 'GaussianSVM', X: np.ndarray, y: np.ndarray, 
                              save_path: Optional[str] = None) -> None:
    """
    绘制分类报告可视化
    
    Args:
        svm: 训练好的GaussianSVM模型
        X: 测试特征
        y: 测试标签
        save_path: 保存路径
    """
    report = svm.get_classification_report(X, y)
    cm = report['confusion_matrix']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 混淆矩阵
    confusion_matrix = np.array([[cm['TN'], cm['FP']], 
                                [cm['FN'], cm['TP']]])
    
    im = ax1.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    ax1.figure.colorbar(im, ax=ax1)
    
    # 添加文本标注
    thresh = confusion_matrix.max() / 2.
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    ax1.set_ylabel('真实标签')
    ax1.set_xlabel('预测标签')
    ax1.set_title('混淆矩阵')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['负类', '正类'])
    ax1.set_yticklabels(['负类', '正类'])
    
    # 性能指标条形图
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    scores = [report['accuracy'], report['precision'], report['recall'], report['f1_score']]
    
    bars = ax2.bar(metrics, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('分数')
    ax2.set_title('分类性能指标')
    
    # 添加数值标注
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分类报告图已保存到: {save_path}")
    
    plt.show()
