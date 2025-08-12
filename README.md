# 🎯 Gaussian SVM from Scratch

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Linyi1225/svm.svg)](https://github.com/Linyi1225/svm/stargazers)

> 纯Python实现的高斯核支持向量机，让你真正理解SVM算法原理，而不是只会调用sklearn！

## 🌟 为什么要重新发明轮子？

- **🔍 透明化学习**：每一行代码都能看懂，不再是黑盒调包
- **🎓 教育价值**：完整的SMO算法实现，最佳的SVM学习材料
- **🛠️ 可定制性**：想改核函数？想调优化策略？随你折腾
- **📊 可视化丰富**：训练过程、决策边界、支持向量，一目了然

## ✨ 核心特性

- ✅ **SMO算法实现**：手写序列最小优化，理解SVM求解过程
- ✅ **高斯核函数**：支持可调参数的RBF核
- ✅ **实时监控**：训练过程可视化，观察收敛情况
- ✅ **完整评估**：精确率、召回率、F1分数、混淆矩阵
- ✅ **实战案例**：广告检测场景，不只是玩具数据
- ✅ **性能对比**：与sklearn SVM全面对比

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/Linyi1225/svm.git
cd svm
pip install -r requirements.txt
```

### 基础使用

```python
from svm import GaussianSVM
import numpy as np

# 创建数据
X = np.random.randn(100, 2)
y = np.random.choice([-1, 1], 100)

# 训练SVM
svm = GaussianSVM(C=1.0, gamma=0.5)
svm.fit(X, y)

# 预测
predictions = svm.predict(X_test)
accuracy = svm.score(X_test, y_test)
```

### 广告检测示例

```python
from svm import GaussianSVM
from svm.datasets import generate_ad_detection_data
from svm.visualizer import plot_decision_boundary

# 生成广告检测数据
X, y = generate_ad_detection_data(n_samples=300)

# 训练模型
svm = GaussianSVM(C=5.0, gamma=0.8)
svm.fit(X, y)

# 可视化结果
plot_decision_boundary(svm, X, y)
print(f"准确率: {svm.score(X, y):.3f}")
```


## 🎯 实战案例：广告检测

我们用4个特征来识别广告和正常内容：

- **点击率**：广告通常有更高的点击率
- **停留时间**：用户在广告页面停留时间较短
- **关键词密度**：广告内容关键词堆砌严重
- **价格敏感度**：广告通常包含价格、优惠信息

```python
# 典型的广告特征
ad_features = [0.12, 30, 0.15, 0.8]  # 高点击率，短停留，高关键词密度，高价格敏感度

# 预测结果
prediction = svm.predict([ad_features])
confidence = svm.decision_function([ad_features])

print(f"预测结果: {'广告' if prediction[0] > 0 else '正常内容'}")
print(f"置信度: {abs(confidence[0]):.3f}")
```

## 📚 算法原理

### SMO算法核心思想

1. **选择优化变量对**：每次选择两个拉格朗日乘子进行优化
2. **KKT条件检查**：判断当前解是否满足最优性条件
3. **二次规划求解**：在约束条件下求解二变量二次规划问题
4. **迭代更新**：重复上述过程直到收敛

### 高斯核函数

```
K(x_i, x_j) = exp(-γ ||x_i - x_j||²)
```

其中γ控制核函数的"宽度"：
- γ大：决策边界复杂，容易过拟合
- γ小：决策边界平滑，可能欠拟合

## 🔧 API参考

### GaussianSVM类

```python
class GaussianSVM:
    def __init__(self, C=1.0, gamma=1.0, tol=1e-3, max_iter=1000):
        """
        参数:
        - C: 正则化参数，控制对误分类的惩罚
        - gamma: 高斯核参数，控制决策边界复杂度
        - tol: 收敛容忍度
        - max_iter: 最大迭代次数
        """
```

### 主要方法

- `fit(X, y)`: 训练模型
- `predict(X)`: 预测类别
- `decision_function(X)`: 获取决策函数值
- `score(X, y)`: 计算准确率
- `get_classification_report(X, y)`: 获取详细评估报告

## 📁 项目结构

```
svm/
├── svm/
│   ├── __init__.py           # 包初始化
│   ├── core.py              # GaussianSVM核心实现
│   ├── visualizer.py        # 可视化工具
│   └── datasets.py          # 数据生成器
├── examples/
│   ├── test.py       # 基础使用示例
│   ├── ad_detection.py      # 广告检测完整示例
│   
├── docs/
│   ├── algorithm.md         # 算法详细原理
│   └── images/             # 文档图片
└── tests/                  # 单元测试
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork这个项目
2. 创建你的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 待办事项

- [ ] 添加更多核函数（多项式核、sigmoid核）
- [ ] 实现多分类SVM（一对一、一对多）
- [ ] 添加特征选择功能
- [ ] 支持在线学习
- [ ] GPU加速版本

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- 感谢《统计学习方法》提供的理论基础
- 感谢所有为开源社区贡献的开发者

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！
