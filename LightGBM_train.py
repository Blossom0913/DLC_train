import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_load import load_mouse_data

def main():
    # Load data
    feature_path = r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\feature8_58.xlsx"
    label_path = r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\merged_labels.xlsx"
    X_raw, y_raw = load_mouse_data(feature_path, label_path)

    # Ensure X_raw and y_raw have the same length
    if len(X_raw) != len(y_raw):
        min_len = min(len(X_raw), len(y_raw))
        print(f"Warning: X_raw and y_raw have different lengths ({len(X_raw)} vs {len(y_raw)}). Trimming to {min_len}.")
        X_raw = X_raw[:min_len]
        y_raw = y_raw[:min_len]

    # Standardize features (optional, LightGBM is robust to scale)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Split data (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_raw, test_size=0.3, random_state=42, stratify=y_raw)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train LightGBM
    params = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'max_depth': 10,
        'boosting_type': 'gbdt',  # or try 'dart'
        'is_unbalance': True,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    # Print best iteration when early stopping is triggered
    print(f"Early stopping at iteration: {model.best_iteration}")

    # Predict and evaluate using the best iteration
    y_pred = np.argmax(model.predict(X_test, num_iteration=model.best_iteration), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test F1 Score: {f1*100:.2f}%")
    print("\nClassification Report (test):")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    main()