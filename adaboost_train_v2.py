import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

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

    # Optionally, remove the base class for a 3-class experiment
    include_base = True  # Set to False for 3-class
    if include_base:
        print("\n=== 4-class experiment (base, aggression, social, nonsocial) ===")
        mask = np.isin(y_raw, [0, 1, 2, 3])
        num_classes = 4
    else:
        print("\n=== 3-class experiment (aggression, social, nonsocial) ===")
        mask = np.isin(y_raw, [1, 2, 3])
        num_classes = 3
    X = X_raw[mask]
    y = y_raw[mask]
    if not include_base:
        y = y - 1  # aggression->0, social->1, nonsocial->2

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Compute class weights for imbalanced data
    classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weight = np.array([class_weights[label] for label in y_train])

    # Define AdaBoost model (with decision tree base estimator)
    base_estimator = DecisionTreeClassifier(max_depth=3)
    clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, learning_rate=0.5, random_state=42)

    # Train
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    # Validation
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Validation F1 Score: {val_f1*100:.2f}%")

    # Test
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1 Score: {test_f1*100:.2f}%")

    # Per-class accuracy
    print("Per-class accuracy (test):")
    for cls in range(num_classes):
        cls_mask = y_test == cls
        cls_acc = 100 * np.sum(y_test_pred[cls_mask] == cls) / np.sum(cls_mask) if np.sum(cls_mask) > 0 else 0.0
        print(f"  Class {cls + (0 if include_base else 1)}: {cls_acc:.2f}%")

    print("\nClassification Report (test):")
    print(classification_report(y_test, y_test_pred, digits=4))

if __name__ == "__main__":
    main()