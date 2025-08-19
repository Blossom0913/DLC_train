from __future__ import print_function
import sys
import os
import traceback
import numpy as np


def check_environment():
    # Require Python 3 for PyTorch
    if sys.version_info[0] < 3:
        print("[SKIP] Python 3 is required to run LSTM.py (found Python {}.{}).".format(sys.version_info[0], sys.version_info[1]))
        return False
    try:
        import torch  # noqa: F401
    except Exception as exc:
        print("[SKIP] PyTorch is not installed: {}".format(exc))
        print("       Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return False
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        print("[SKIP] scikit-learn is not installed: {}".format(exc))
        print("       Install with: pip install scikit-learn")
        return False
    return True


def make_synthetic_data(num_samples=2000, num_features=8, num_classes=4, seed=123):
    rng = np.random.RandomState(seed)
    X = rng.randn(num_samples, num_features).astype(np.float32)
    # Create non-trivial labels
    logits = np.stack([
        0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.2 * rng.randn(num_samples),
        -0.3 * X[:, 0] + 0.9 * X[:, 2] + 0.2 * rng.randn(num_samples),
        0.4 * X[:, 3] - 0.7 * X[:, 4] + 0.2 * rng.randn(num_samples),
        0.6 * X[:, 5] + 0.6 * X[:, 6] - 0.3 * X[:, 7] + 0.2 * rng.randn(num_samples),
    ], axis=1)
    y = np.argmax(logits, axis=1).astype(np.int64)
    return X, y


def main():
    if not check_environment():
        # Exit with code 0 to indicate the test was skipped, not failed
        sys.exit(0)

    # Imports that require Python 3 and installed deps
    import torch
    from LSTM import run_experiment

    # Use CPU for portability
    device = torch.device('cpu')

    # Small config for a quick smoke test
    CONFIG = {
        'batch_size': 64,
        'epochs': 3,
        'device': device,
    }

    # Synthetic data
    X_raw, y_raw = make_synthetic_data(num_samples=1200, num_features=8, num_classes=4, seed=123)

    print("[INFO] Starting LSTM smoke test (synthetic data, {} epochs)...".format(CONFIG['epochs']))
    try:
        history, model_path = run_experiment(
            X_raw, y_raw,
            experiment_name="lstm_test",
            num_classes=4,
            include_base=True,
            **CONFIG
        )
        # Basic sanity checks
        assert isinstance(history, dict), "history should be a dict"
        for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            assert key in history, "history missing key '{}'".format(key)
        print("[PASS] LSTM run_experiment executed successfully.")
        print("[PASS] Model saved to: {}".format(model_path))

    except Exception:
        print("[FAIL] LSTM run_experiment raised an exception:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


