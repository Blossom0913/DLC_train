# DeepLabVideo: Mouse Behavior Classification

This repository contains code for training, evaluating, and visualizing machine learning models for mouse social behavior classification. It includes CNN, LSTM, GMM, and LightGBM baselines, plus experiment runners, academic-style plots, and utilities for data preparation.

- Video：https://youtu.be/oTXjbmTi8IQ
- Dataset DOI: https://doi.org/10.6084/m9.figshare.30393298
- OS/Env: Developed and tested on Windows with Python. Works in VS Code.

## Dataset

Download the dataset from the Figshare DOI above. The experiments expect two Excel files for the 58-video set:

- `dataset58/feature8_58.xlsx` — features matrix (8 columns)
- `dataset58/merged_labels_aggression.xlsx` — labels vector (one column)

Place these files under the project root, e.g.:

```
DLC_train/
  dataset58/
    feature8_58.xlsx
    merged_labels_aggression.xlsx
```

If your files live elsewhere, update the paths passed to `load_mouse_data` in the scripts.

## Project structure

Key files and folders:

- `model_comparison_aggression_multiclass.py` — Run multi-class aggression experiments across models with error bars and figures.
- `model_comparison_with_error_bars.py` — General model comparison with error bars (non-aggression specific).
- `CNN.py`, `LSTM.py`, `GMM.py`, `LightGBM.py` — Individual model training/evaluation entrypoints (`run_experiment`).
- `data_solver.py`, `data_load.py` — Data loading and preprocessing utilities (sequence creation, scaling).
- `graph_samples/` — Plotting scripts and generated figures.
  - `plot_label_proportions.py` — Plot base/social/nonsocial/aggression proportions.
  - `plot_aggression_label_proportions.py` — Plot detailed aggression-label proportions in academic style.
  - Saved images like `model_comparison_aggression_multiclass_overall.png`, etc.
- `model_comparison_aggression/` — A self-contained subproject variant (includes its own `requirements.txt` and `README.md`).

## Installation

Python 3.9–3.11 recommended.

Option A: Use the provided requirements from the subproject:

```
pip install -r model_comparison_aggression/requirements.txt
```

Option B: Minimal manual install (example):

```
pip install numpy pandas scikit-learn matplotlib seaborn scipy
pip install lightgbm
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or your CUDA build
```

## Quick start

1) Multi-class aggression comparison (excludes non-aggression/base):

```
python model_comparison_aggression_multiclass.py
```

This script:
- Loads features + labels, trims length if mismatched, filters out non-aggression labels, maps classes
- Runs multiple repetitions for each model (CNN, LSTM, GMM, LightGBM) to compute mean/std
- Saves figures in the working directory:
  - `model_comparison_aggression_multiclass_overall.png`
  - `model_comparison_aggression_multiclass_per_class.png`
  - `model_comparison_aggression_multiclass_best_worst.png`
  - `model_comparison_aggression_multiclass_stability.png`

2) Individual models (example: LightGBM):

```
python LightGBM.py
```

Each model script prints Accuracy / F1 metrics and saves a checkpoint in the current directory. CNN/LSTM save `.pth`; LightGBM saves a `.pkl` (model + scaler).

3) Academic-style plots:

- Overall label proportions (base, aggression, social, nonsocial):
  - `graph_samples/plot_label_proportions.py`
- Aggression-only label proportions with better spacing:
  - `graph_samples/plot_aggression_label_proportions.py`

Run either via VS Code or terminal, e.g.:

```
python graph_samples/plot_label_proportions.py
python graph_samples/plot_aggression_label_proportions.py
```

The scripts save PNG/PDF/SVG (where applicable).

## Label schema

Top-level behavioral labels:

```
{'base': 0, 'aggression': 1, 'social': 2, 'nonsocial': 3}
```

Aggression labels (kept for multi-class aggression experiments; non-aggression/base removed before training):

```
{
  'freezing': 2,
  'lateralthreat': 3,
  'keepdown': 4,
  'chase': 5,
  'uprightposture': 6,
  'bite': 8,
  'clinch': 10
}
```

The `model_comparison_aggression_multiclass.py` script filters out non-aggression samples and maps the remaining non-contiguous labels to a contiguous range before model training and evaluation.

## Reproducibility

- The comparison script repeats runs with varied seeds for error bars. Within a run, seeds are set for NumPy and PyTorch (if available).
- Scaling is fit on train only for deep models to avoid leakage; LightGBM scales features as a convenience (it is robust to feature scale).

## Outputs

- Model checkpoints are saved in the working directory with timestamps, e.g. `cnn_cnn_4class_YYYYMMDD_HHMMSS.pth`, `lightgbm_*.pkl`.
- Comparison figures are saved as PNG files.
- Plot scripts output high-resolution publication-ready figures.

## Troubleshooting

- Boolean mask mismatch after filtering: the script trims `X_raw`/`y_raw` lengths if unequal. Ensure your feature and label files cover the same time span/rows.
- Path issues on Windows: prefer absolute paths or keep the `dataset58/` folder under this project root.
- GPU vs CPU: PyTorch device is configurable; defaults to CPU in scripts. Switch to CUDA if available.

## How to cite

If you use this code or dataset in academic work, please cite the dataset DOI and this repository:

- Dataset: https://doi.org/10.6084/m9.figshare.30393298
- Code: add your preferred software citation (e.g., Zenodo DOI if archived)

Example citation format:

```
Author(s). DeepLabVideo: Mouse Behavior Classification (Version YYYY.MM). Repository name. URL
Dataset: Figshare. DOI: 10.6084/m9.figshare.30393298
```

## License

Specify your license here (e.g., MIT). If none is provided, all rights reserved by default.
