"""
train_baselines.py
------------------
Purpose:
    - Command-line (or Run Configuration) entry point to train all classical baselines.
    - Loads data/training.pkl (or .npz/.joblib), flattens windows for ML,
      trains models, prints a metric table, and saves:
        * artifacts/baseline_results.csv
        * artifacts/cm_<model>.png

Run in PyCharm:
    - Right-click this file -> Run 'train_baselines'
    - Or set PYTHONPATH to project root if needed.

If training data is missing:
    - The script prints a clear message asking teammates for the export.
"""

# --- make 'src' importable when running from terminal ---
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------------------------

from src.models.io_adapters import load_training, flatten_windows
from src.models.baselines import train_all_models


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def save_conf_mats(conf_mats, out_dir: str):
    """
    Save each confusion matrix as a simple PNG so you can drop it into slides.
    (We avoid seaborn to keep dependencies simple.)
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, cm in conf_mats.items():
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix – {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        # write counts in each cell
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_{name}.png"))
        plt.close()


def main(args):
    train_path = args.train_path
    out_dir = args.out_dir

    if not os.path.exists(train_path):
        print(f"❗ Missing {train_path}. Ask M1/M2 to export training data "
              f"(X_seg, y_exercise, optional groups).")
        return

    # 1) Load the sequences + labels
    X_seq, y, _groups = load_training(train_path)

    # 2) Flatten to 2D for classical ML models (we keep X_seq for DL later)
    X = flatten_windows(X_seq)

    # 3) Train all baselines and collect metrics + confusion matrices
    results, cms = train_all_models(X, y)

    # 4) Save artifacts for your presentation
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, "baseline_results.csv")
    results.to_csv(results_csv, index=False)
    print("\n=== Baseline Results (saved to artifacts/baseline_results.csv) ===")
    print(results)

    save_conf_mats(cms, out_dir)
    print(f"\nSaved confusion matrices to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/training.pkl",
                        help="Path to training pickle/npz/joblib export")
    parser.add_argument("--out_dir", default="artifacts",
                        help="Directory to write results and images")
    main(parser.parse_args())
