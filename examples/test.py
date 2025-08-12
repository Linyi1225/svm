"""
GaussianSVM基础使用示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svm import GaussianSVM, generate_toy_data, plot_decision_boundary, plot_training_history
from sklearn.model_selection import train_test_split

def main():
    print("🎯 GaussianSVM 基础使用示例")
    print("=" * 40)
    
    # 生成数据
    print("📊 生成玩具数据集...")
    X, y = generate_toy_data(n_samples=200, noise=0.1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 创建和训练SVM
    print("\n🚀 训练GaussianSVM...")
    svm = GaussianSVM(C=1.0, gamma=0.5, max_iter=200)
    svm.fit(X_train, y_train)
    
    # 评估性能
    print("\n📈 模型评估:")
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    
    print(f"训练准确率: {train_acc:.3f}")
    print(f"测试准确率: {test_acc:.3f}")
    
    # 获取详细报告
    report = svm.get_classification_report(X_test, y_test)
    print(f"精确率: {report['precision']:.3f}")
    print(f"召回率: {report['recall']:.3f}")
    print(f"F1分数: {report['f1_score']:.3f}")
    
    # 可视化
    print("\n📊 生成可视化图表...")
    plot_decision_boundary(svm, X_train, y_train)
    plot_training_history(svm)
    
    # 预测新样本
    print("\n🔍 预测新样本:")
    new_samples = [[0, 0], [3, 3], [-3, -3]]
    predictions = svm.predict(new_samples)
    probabilities = svm.predict_proba(new_samples)
    
    for i, (sample, pred, prob) in enumerate(zip(new_samples, predictions, probabilities)):
        print(f"样本 {sample}: 预测={pred:2.0f}, 概率=[{prob[0]:.3f}, {prob[1]:.3f}]")

if __name__ == "__main__":
    main()
