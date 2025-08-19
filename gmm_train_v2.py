import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import joblib
from data_load import *

def preprocess_data(X, y, window_length=10):
    """预处理数据并展平时间窗口"""
    X_windowed = []
    y_windowed = []
    for i in range(len(X) - window_length + 1):
        X_windowed.append(X[i:i+window_length])
        y_windowed.append(y[i+window_length-1])
    
    if len(X_windowed) == 0:
        raise ValueError("窗口长度大于输入数据长度")
    if X_windowed[0].shape != (window_length, X.shape[1]):
        raise ValueError(f"窗口形状异常，期望: {(window_length, X.shape[1])}，实际: {X_windowed[0].shape}")
    
    # 展平窗口数据为特征向量
    X_flattened = np.array(X_windowed).reshape(len(X_windowed), -1)
    return X_flattened, np.array(y_windowed)

class GMMClassifier:
    """多类别GMM分类器"""
    def __init__(self, n_components=3, covariance_type='full'):
        self.models = {}
        self.n_components = n_components
        self.covariance_type = covariance_type
        
    def fit(self, X, y):
        """为每个类别训练独立的GMM模型"""
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            class_data = X[y == cls]
            if len(class_data) < self.n_components:
                print(f"警告: 类别{cls}样本数不足，已自动调整n_components")
                n_comp = min(len(class_data), self.n_components)
                gmm = GaussianMixture(n_components=n_comp, 
                                    covariance_type=self.covariance_type)
            else:
                gmm = GaussianMixture(n_components=self.n_components,
                                    covariance_type=self.covariance_type)
            gmm.fit(class_data)
            self.models[cls] = gmm
            
    def predict(self, X):
        """基于对数概率进行预测"""
        log_probs = []
        for cls in self.classes_:
            try:
                log_prob = self.models[cls].score_samples(X)
            except:
                log_prob = np.full(X.shape[0], -np.inf)
            log_probs.append(log_prob)
        return self.classes_[np.argmax(log_probs, axis=0)]

def plot_and_save_training_curves(history, save_path='gmm_training_curves_8.png'):
    """绘制训练曲线（含F1分数）"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Folds')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Training F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score Curves')
    plt.xlabel('Folds')
    plt.ylabel('F1 Score (%)')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"训练曲线已保存至: {save_path}")

def evaluate_model(model, X, y):
    """评估模型性能"""
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds) * 100
    f1 = f1_score(y, preds, average='weighted') * 100
    return accuracy, f1

def calculate_class_accuracy(y_true, y_pred, num_classes=5):
    """计算每个类别的准确率"""
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        class_correct[true] += (true == pred)
    
    return [100 * class_correct[i]/class_total[i] if class_total[i] > 0 else 0.0 
            for i in range(num_classes)]


def run_experiment(X_raw, y_raw, window_length, test_size, n_components, n_splits, include_base=True):
    if include_base:
        print("\n=== 4-class experiment (base, aggression, social, nonsocial) ===")
        mask = np.ones_like(y_raw, dtype=bool)
        num_classes = 4
    else:
        print("\n=== 3-class experiment (aggression, social, nonsocial) ===")
        mask = y_raw != 0  # Remove base class
        num_classes = 3

    X_raw_exp = X_raw[mask]
    y_raw_exp = y_raw[mask]

    # Train/test split
    X_train_raw, X_test_raw = X_raw_exp[:-test_size], X_raw_exp[-test_size:]
    y_train_raw, y_test_raw = y_raw_exp[:-test_size], y_raw_exp[-test_size:]

    # Windowing
    X_train_windows, y_train_windows = preprocess_data(X_train_raw, y_train_raw, window_length)
    X_test_windows, y_test_windows = preprocess_data(X_test_raw, y_test_raw, window_length)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_windows)
    X_test_scaled = scaler.transform(X_test_windows)

    # Train model
    model = GMMClassifier(n_components=n_components)
    model.fit(X_train_scaled, y_train_windows)

    # Evaluate
    test_acc, test_f1 = evaluate_model(model, X_test_scaled, y_test_windows)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1 Score: {test_f1:.2f}%")


        # Per-class accuracy
    y_pred = model.predict(X_test_scaled)
    # Remap labels for 3-class experiment
    if not include_base:
        y_test_windows = y_test_windows - 1
        y_pred = y_pred - 1
    class_acc = calculate_class_accuracy(y_test_windows, y_pred, num_classes)
    print("Per-class accuracy:")
    for cls, acc in enumerate(class_acc):
        print(f"  Class {cls + (0 if include_base else 1)}: {acc:.2f}%")

    # Per-class accuracy
    y_pred = model.predict(X_test_scaled)
    class_acc = calculate_class_accuracy(y_test_windows, y_pred, num_classes)
    print("Per-class accuracy:")
    for cls, acc in enumerate(class_acc):
        print(f"  Class {cls + (0 if include_base else 1)}: {acc:.2f}%")


def main():
    # 参数配置
    WINDOW_LENGTH = 10
    N_SPLITS = 3
    GMM_COMPONENTS = 10
    model_name = 'gmm_model_8.pkl'
    
    # 数据加载
    feature_path = r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\feature8_58.xlsx"
    label_path = r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\merged_labels.xlsx"
    X_raw, y_raw = load_mouse_data(feature_path, label_path)

    # After loading X_raw, y_raw
    if len(X_raw) != len(y_raw):
        min_len = min(len(X_raw), len(y_raw))
        print(f"Warning: X_raw and y_raw have different lengths ({len(X_raw)} vs {len(y_raw)}). Trimming to {min_len}.")
        X_raw = X_raw[:min_len]
        y_raw = y_raw[:min_len]
    
    # 划分数据集
    test_size = 40000
    X_train_raw, X_test_raw = X_raw[:-test_size], X_raw[-test_size:]
    y_train_raw, y_test_raw = y_raw[:-test_size], y_raw[-test_size:]
    
    # 窗口化处理
    X_train_windows, y_train_windows = preprocess_data(X_train_raw, y_train_raw, WINDOW_LENGTH)
    X_test_windows, y_test_windows = preprocess_data(X_test_raw, y_test_raw, WINDOW_LENGTH)

    # 交叉验证
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    history = {'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}

    # Run 4-class experiment
    run_experiment(X_raw, y_raw, WINDOW_LENGTH, test_size, GMM_COMPONENTS, N_SPLITS, include_base=True)

    # Run 3-class experiment
    run_experiment(X_raw, y_raw, WINDOW_LENGTH, test_size, GMM_COMPONENTS, N_SPLITS, include_base=False)


    # for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_windows)):
    #     print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")
        
    #     # 数据准备
    #     X_train_fold, y_train_fold = X_train_windows[train_idx], y_train_windows[train_idx]
    #     X_val_fold, y_val_fold = X_train_windows[val_idx], y_train_windows[val_idx]
        
    #     # 标准化
    #     scaler = StandardScaler()
    #     X_train_fold = scaler.fit_transform(X_train_fold)
    #     X_val_fold = scaler.transform(X_val_fold)
        
    #     # 训练模型
    #     model = GMMClassifier(n_components=GMM_COMPONENTS)
    #     model.fit(X_train_fold, y_train_fold)
        
    #     # 评估
    #     train_acc, train_f1 = evaluate_model(model, X_train_fold, y_train_fold)
    #     val_acc, val_f1 = evaluate_model(model, X_val_fold, y_val_fold)
        
    #     # 记录结果
    #     history['train_acc'].append(train_acc)
    #     history['val_acc'].append(val_acc)
    #     history['train_f1'].append(train_f1)
    #     history['val_f1'].append(val_f1)
        
    #     # 打印类别准确率
    #     y_pred = model.predict(X_val_fold)
    #     class_acc = calculate_class_accuracy(y_val_fold, y_pred)
    #     print(f"Fold {fold+1} 验证集各类别准确率:")
    #     for cls, acc in enumerate(class_acc):
    #         print(f"  类别 {cls}: {acc:.2f}%")

    # # 全量训练
    # final_scaler = StandardScaler()
    # X_train_scaled = final_scaler.fit_transform(X_train_windows)
    # final_model = GMMClassifier(n_components=GMM_COMPONENTS)
    # final_model.fit(X_train_scaled, y_train_windows)
    
    # # 测试集评估
    # X_test_scaled = final_scaler.transform(X_test_windows)
    # test_acc, test_f1 = evaluate_model(final_model, X_test_scaled, y_test_windows)
    # print(f"\n最终测试集结果:")
    # print(f"  准确率: {test_acc:.2f}%")
    # print(f"  F1分数: {test_f1:.2f}%")
    
    # # 保存模型
    # joblib.dump({'model': final_model, 'scaler': final_scaler}, model_name)
    # plot_and_save_training_curves(history)
    # print(f"模型已保存至: {model_name}")

if __name__ == "__main__":
    main()