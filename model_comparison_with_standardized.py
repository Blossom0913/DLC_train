import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Use the experiment runners that now standardize with train-fitted scalers
from GMM import run_experiment as run_gmm_experiment
from LSTM import run_experiment as run_lstm_experiment
from CNN import run_experiment as run_cnn_experiment
from LightGBM import run_experiment as run_lightgbm_experiment
from data_load import load_mouse_data


def summarize_metrics(name, result):
    metrics = result['test_metrics']
    compact_keys = ['accuracy', 'weighted_f1', 'macro_f1', 'best_class_f1', 'worst_class_f1']
    compact = {k: float(metrics.get(k, 0)) for k in compact_keys}
    print(f"\n{name} (standardized) summary:")
    print(compact)


def main():
    print("=== Model Comparison (Standardized, train-fitted) ===")

    # Load raw data
    X_raw, y_raw = load_mouse_data(
        r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\feature8_58.xlsx",
        r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\merged_labels.xlsx"
    )

    # Focus on 4-class (include base) by default; adjust as needed
    include_base = True
    num_classes = 4 if include_base else 3

    results = {}

    # CNN
    results['CNN'] = run_cnn_experiment(
        X_raw, y_raw, batch_size=256, epochs=50, device='cpu',
        num_classes=num_classes, include_base=include_base, experiment_name='cnn_std'
    )
    summarize_metrics('CNN', results['CNN'])

    # LSTM
    results['LSTM'] = run_lstm_experiment(
        X_raw, y_raw, batch_size=256, epochs=50, device='cpu',
        num_classes=num_classes, include_base=include_base, experiment_name='lstm_std'
    )
    summarize_metrics('LSTM', results['LSTM'])

    # GMM
    results['GMM'] = run_gmm_experiment(
        X_raw, y_raw, n_components=10,
        num_classes=num_classes, include_base=include_base, experiment_name='gmm_std'
    )
    summarize_metrics('GMM', results['GMM'])

    # LightGBM
    results['LightGBM'] = run_lightgbm_experiment(
        X_raw, y_raw,
        num_classes=num_classes, include_base=include_base, experiment_name='lightgbm_std'
    )
    summarize_metrics('LightGBM', results['LightGBM'])

    print("\n✓ Completed standardized model comparison.")


if __name__ == "__main__":
    main()

