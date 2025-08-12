"""
Gaussian SVM核心实现
使用SMO算法求解SVM对偶问题
"""

import numpy as np
import time
from typing import Optional, Dict, List, Tuple

class GaussianSVM:
    """
    高斯核支持向量机实现
    
    使用SMO算法求解SVM对偶问题
    """
    
    def __init__(self, C: float = 1.0, gamma: float = 1.0, 
                 tol: float = 1e-3, max_iter: int = 1000):
        """
        初始化GaussianSVM
        
        Args:
            C: 正则化参数，控制对误分类的惩罚
            gamma: 高斯核参数，控制核函数的宽度
            tol: 收敛容忍度
            max_iter: 最大迭代次数
        """
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.training_time = 0
        self.training_history: List[Dict] = []
        
    def gaussian_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        高斯核函数: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
        
        Args:
            x1: 输入数据1
            x2: 输入数据2
            
        Returns:
            核矩阵
        """
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)

        if x1.shape[1] != x2.shape[1]:
            raise ValueError(f"特征维度不匹配: {x1.shape[1]} vs {x2.shape[1]}")

        sq_dists = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * sq_dists)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianSVM':
        """
        训练SVM模型
        
        Args:
            X: 训练特征，形状为 (n_samples, n_features)
            y: 训练标签，形状为 (n_samples,)，值为 -1 或 1
            
        Returns:
            self: 训练好的模型
        """
        start_time = time.time()
        
        self.X = X.copy()
        self.y = y.copy()
        n_samples, n_features = X.shape

        print(f"开始训练SVM: {n_samples}个样本, {n_features}个特征")
        
        # 计算核矩阵
        print("计算核矩阵...")
        self.K = self.gaussian_kernel(X, X)

        # 初始化拉格朗日乘子
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        self.training_history = []

        # SMO算法主循环
        print("开始SMO优化...")
        for iteration in range(self.max_iter):
            alpha_prev = self.alpha.copy()
            num_changed_alphas = 0

            for i in range(n_samples):
                E_i = self._decision_function_single(i) - y[i]

                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):

                    j = self._select_j(i, E_i, n_samples)
                    if j == i:
                        continue
                        
                    E_j = self._decision_function_single(j) - y[j]
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= (y[j] * (E_i - E_j)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                        
                    num_changed_alphas += 1

            # 记录训练进度
            current_accuracy = self.score(X, y)
            self.training_history.append({
                'iteration': iteration,
                'accuracy': current_accuracy,
                'num_sv': np.sum(self.alpha > 1e-5),
                'alpha_changes': num_changed_alphas
            })
            
            if iteration % 50 == 0:
                print(f"第{iteration}轮: 准确率={current_accuracy:.3f}, 支持向量={np.sum(self.alpha > 1e-5)}")

            if np.sum(np.abs(self.alpha - alpha_prev)) < self.tol:
                print(f"在第{iteration}轮收敛!")
                break

        # 保存支持向量
        sv_idx = self.alpha > 1e-5
        self.support_vectors_ = X[sv_idx]
        self.support_vector_labels_ = y[sv_idx]
        self.alpha_sv = self.alpha[sv_idx]
        
        self.training_time = time.time() - start_time
        
        print(f"训练完成! 用时{self.training_time:.2f}秒")
        print(f"共找到 {len(self.support_vectors_)} 个支持向量 ({len(self.support_vectors_)/len(X)*100:.1f}%)")
        return self

    def _decision_function_single(self, i: int) -> float:
        """计算单个样本的决策函数值"""
        return np.sum(self.alpha * self.y * self.K[i, :]) + self.b

    def _select_j(self, i: int, E_i: float, n_samples: int) -> int:
        """选择第二个优化变量j"""
        max_E_diff = 0
        best_j = i
        
        boundary_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        
        if len(boundary_indices) > 1:
            for k in boundary_indices:
                if k == i:
                    continue
                E_k = self._decision_function_single(k) - self.y[k]
                if abs(E_i - E_k) > max_E_diff:
                    max_E_diff = abs(E_i - E_k)
                    best_j = k
        
        if best_j == i:
            candidates = list(range(n_samples))
            candidates.remove(i)
            best_j = np.random.choice(candidates)
            
        return best_j

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算决策函数值"""
        K_test = self.gaussian_kernel(X, self.X)
        return np.sum(self.alpha * self.y * K_test, axis=1) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        return np.sign(self.decision_function(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        decision_scores = self.decision_function(X)
        probabilities = 1 / (1 + np.exp(-decision_scores))
        return np.column_stack([1 - probabilities, probabilities])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_classification_report(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """获取详细的分类报告"""
        predictions = self.predict(X)
        
        tp = np.sum((predictions == 1) & (y == 1))
        tn = np.sum((predictions == -1) & (y == -1))
        fp = np.sum((predictions == 1) & (y == -1))
        fn = np.sum((predictions == -1) & (y == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': (tp + tn) / len(y),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        }
