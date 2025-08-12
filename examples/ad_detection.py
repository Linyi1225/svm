"""
广告检测完整示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svm import GaussianSVM, generate_ad_detection_data, plot_decision_boundary, plot_classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def main():
    print("🎯 广告检测SVM示例")
    print("=" * 40)
    
    # 生成广告检测数据
    print("📊 生成广告检测数据集...")
    X, y = generate_ad_detection_data(n_samples=400)
    
    feature_names = ['点击率', '停留时间', '关键词密度', '价格敏感度']
    print(f"特征: {feature_names}")
    print(f"广告样本: {np.sum(y == 1)}")
    print(f"正常内容: {np.sum(y == -1)}")
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 训练SVM
    print("\n🚀 训练广告检测SVM...")
    svm = GaussianSVM(C=5.0, gamma=0.8, max_iter=300)
    svm.fit(X_train, y_train)
    
    # 评估性能
    print("\n📈 性能评估:")
    train_report = svm.get_classification_report(X_train, y_train)
    test_report = svm.get_classification_report(X_test, y_test)
    
    print("训练集表现:")
    print(f"  准确率: {train_report['accuracy']:.3f}")
    print(f"  精确率: {train_report['precision']:.3f}")
    print(f"  召回率: {train_report['recall']:.3f}")
    print(f"  F1分数: {train_report['f1_score']:.3f}")
    
    print("测试集表现:")
    print(f"  准确率: {test_report['accuracy']:.3f}")
    print(f"  精确率: {test_report['precision']:.3f}")
    print(f"  召回率: {test_report['recall']:.3f}")
    print(f"  F1分数: {test_report['f1_score']:.3f}")
    
    # 可视化（使用前两个特征）
    print("\n📊 生成可视化...")
    plot_decision_boundary(svm, X_train, y_train)
    plot_classification_report(svm, X_test, y_test)
    
    # 实际案例预测
    print("\n🔍 实际案例预测:")
    
    # 构造一些真实的测试案例
    test_cases = [
        {
            'name': '可疑广告',
            'features': [0.12, 25, 0.15, 0.85],  # 高点击率，短停留，高关键词密度，高价格敏感度
            'description': '高点击率，短停留时间，高关键词密度，高价格敏感度'
        },
        {
            'name': '优质内容',
            'features': [0.025, 180, 0.01, 0.05],  # 适中点击率，长停留，低关键词密度，低价格敏感度
            'description': '适中点击率，长停留时间，低关键词密度，低价格敏感度'
        },
        {
            'name': '模糊案例',
            'features': [0.06, 80, 0.05, 0.3],  # 中等各项指标
            'description': '各项指标都在中等水平'
        }
    ]
    
    # 注意：需要用相同的标准化器处理新数据
    # 这里我们重新生成数据来获取scaler，在实际使用中应该保存训练时的scaler
    X_original, _ = generate_ad_detection_data(n_samples=400)
    scaler = StandardScaler()
    scaler.fit(X_original)
    
    for case in test_cases:
        # 标准化特征
        features_scaled = scaler.transform([case['features']])
        prediction = svm.predict(features_scaled)[0]
        probabilities = svm.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        result = "广告" if prediction > 0 else "正常内容"
        
        print(f"\n{case['name']}:")
        print(f"  特征: {case['description']}")
        print(f"  原始值: {case['features']}")
        print(f"  预测结果: {result}")
        print(f"  置信度: {confidence:.3f}")

    # 分析模型特征重要性（简单分析）
    print("\n🔬 特征分析:")
    print("通过支持向量分析模型关注的特征模式...")
    
    if hasattr(svm, 'support_vectors_'):
        sv_mean = np.mean(svm.support_vectors_, axis=0)
        sv_std = np.std(svm.support_vectors_, axis=0)
        
        print("支持向量特征统计:")
        for i, feature in enumerate(feature_names):
            print(f"  {feature}: 均值={sv_mean[i]:.3f}, 标准差={sv_std[i]:.3f}")

if __name__ == "__main__":
    main()
