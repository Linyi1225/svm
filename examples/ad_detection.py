"""
广告检测完整示例 - 改进版
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
    
    # 生成广告检测数据（注意：这里生成的数据已经标准化了）
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
    
    print(f"训练集标签分布: 广告={np.sum(y_train == 1)}, 内容={np.sum(y_train == -1)}")
    
    # 训练SVM - 调整参数以获得更好的性能
    print("\n🚀 训练广告检测SVM...")
    svm = GaussianSVM(C=1.0, gamma=0.5, max_iter=300)  # 降低复杂度
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
    
    # 实际案例预测 - 重新生成原始数据并正确处理标准化
    print("\n🔍 实际案例预测:")
    
    # 重新生成原始数据（未标准化）来创建正确的scaler
    print("重新生成原始数据以获取正确的标准化器...")
    X_raw, y_raw = generate_raw_ad_detection_data(n_samples=1000)
    
    # 用原始数据训练标准化器
    scaler = StandardScaler()
    scaler.fit(X_raw)
    
    # 构造测试案例（使用原始特征值）
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
        },
        {
            'name': '典型内容',
            'features': [0.02, 200, 0.005, 0.02],  # 更典型的内容特征
            'description': '低点击率，长停留时间，极低关键词密度，极低价格敏感度'
        }
    ]
    
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
        print(f"  标准化后: {features_scaled[0]}")
        print(f"  预测结果: {result}")
        print(f"  置信度: {confidence:.3f}")
        print(f"  决策分数: {svm.decision_function(features_scaled)[0]:.3f}")

    # 参数敏感性分析
    print("\n🔧 参数敏感性分析:")
    test_params = [
        (0.1, 0.1), (0.1, 1.0), (1.0, 0.1), (1.0, 1.0), (10.0, 0.1), (10.0, 1.0)
    ]
    
    for C, gamma in test_params:
        svm_test = GaussianSVM(C=C, gamma=gamma, max_iter=100)
        svm_test.fit(X_train, y_train)
        acc = svm_test.score(X_test, y_test)
        print(f"  C={C}, γ={gamma}: 测试准确率={acc:.3f}")
    
    # 分析模型特征重要性
    print("\n🔬 特征分析:")
    print("通过支持向量分析模型关注的特征模式...")
    
    if hasattr(svm, 'support_vectors_'):
        sv_mean = np.mean(svm.support_vectors_, axis=0)
        sv_std = np.std(svm.support_vectors_, axis=0)
        
        print("支持向量特征统计 (标准化后):")
        for i, feature in enumerate(feature_names):
            print(f"  {feature}: 均值={sv_mean[i]:.3f}, 标准差={sv_std[i]:.3f}")
        
        # 分析不同类别的支持向量
        pos_sv = svm.support_vectors_[svm.support_vector_labels_ == 1]
        neg_sv = svm.support_vectors_[svm.support_vector_labels_ == -1]
        
        if len(pos_sv) > 0 and len(neg_sv) > 0:
            print("\n正类支持向量特征均值:")
            pos_mean = np.mean(pos_sv, axis=0)
            for i, feature in enumerate(feature_names):
                print(f"  {feature}: {pos_mean[i]:.3f}")
                
            print("负类支持向量特征均值:")
            neg_mean = np.mean(neg_sv, axis=0)
            for i, feature in enumerate(feature_names):
                print(f"  {feature}: {neg_mean[i]:.3f}")

def generate_raw_ad_detection_data(n_samples=300, random_state=42):
    """
    生成原始（未标准化）的广告检测数据
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 正常内容 (非广告, y = -1)
    normal_clickrate = np.random.normal(0.03, 0.01, n_samples//2)
    normal_dwelltime = np.random.normal(120, 30, n_samples//2)
    normal_keyword_density = np.random.normal(0.02, 0.008, n_samples//2)
    normal_price_sensitivity = np.random.normal(0.1, 0.05, n_samples//2)

    # 添加非线性相关性
    quality_boost = np.random.normal(0, 0.5, n_samples//2)
    normal_clickrate += quality_boost * 0.01
    normal_dwelltime += quality_boost * 20

    X_normal = np.column_stack([normal_clickrate, normal_dwelltime,
                               normal_keyword_density, normal_price_sensitivity])
    y_normal = -np.ones(n_samples//2)

    # 广告内容 (广告, y = 1)
    ad_clickrate = np.random.normal(0.08, 0.02, n_samples//2)
    ad_dwelltime = np.random.normal(45, 15, n_samples//2)
    ad_keyword_density = np.random.normal(0.08, 0.02, n_samples//2)
    ad_price_sensitivity = np.random.normal(0.6, 0.15, n_samples//2)

    # 添加广告特有的非线性模式
    promo_factor = np.random.normal(0, 1, n_samples//2)
    ad_keyword_density += np.abs(promo_factor) * 0.03
    ad_price_sensitivity += np.abs(promo_factor) * 0.2

    X_ads = np.column_stack([ad_clickrate, ad_dwelltime,
                            ad_keyword_density, ad_price_sensitivity])
    y_ads = np.ones(n_samples//2)

    # 合并数据
    X = np.vstack([X_normal, X_ads])
    y = np.hstack([y_normal, y_ads])

    # 确保所有特征都是正值，但不标准化
    X = np.clip(X, 0.001, None)
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y

if __name__ == "__main__":
    main()
