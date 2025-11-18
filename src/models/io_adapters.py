"""
io_adapters.py
--------------
Purpose:
    - Provide a small, stable interface for loading the team's exported datasets.
    - Accept .pkl, .joblib, or .npz files for robustness across machines.
    - Return:
         * X_seq: (N, 50, 45) windowed sensor sequences for deep learning
         * y:     (N,) class labels (e.g., exercise id)
         * groups:(N,) optional subject id per window (for LOSO/LOEO)
    - Also expose a helper to flatten sequences for classical ML baselines.

Typical files expected:
    data/training.pkl
    data/test.pkl

Team contract (what M1/M2 export):
    training: {"X_seg": (N,50,45), "y_exercise": (N,), "groups": (N, optional)}
    test:     {"X_test": (M,50,45), "y_test_exercise": (M, optional)}
"""

import os
import pickle
from typing import Tuple, Optional, Dict, Any

import numpy as np

# joblib is optional; if not installed, we just won't support .joblib
try:
    import joblib
except Exception:
    joblib = None


# ---- low-level loaders -------------------------------------------------------

def _load_pickle(path: str) -> Dict[str, Any]:
    """Load a Python pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_joblib(path: str) -> Dict[str, Any]:
    """Load a joblib file (if joblib is installed)."""
    if joblib is None:
        raise RuntimeError("joblib not installed; cannot load .joblib file.")
    return joblib.load(path)


def _load_npz(path: str) -> Dict[str, Any]:
    """
    Load a compressed NumPy archive.
    np.load returns an NpzFile; convert to a normal dict for convenience.
    """
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def smart_load(path: str) -> Dict[str, Any]:
    """
    Pick the right loader based on file extension.
    Supported: .pkl, .joblib, .npz
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        return _load_pickle(path)
    if ext == ".joblib":
        return _load_joblib(path)
    if ext == ".npz":
        return _load_npz(path)
    raise ValueError(f"Unsupported data format: {ext}")


# ---- public adapters ---------------------------------------------------------

def load_training(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load the training dataset.
    Returns:
        X_seq  : ndarray (N, 50, 45)  -> sequences for DL
        y      : ndarray (N,)         -> labels (e.g., exercise id)
        groups : ndarray (N,) or None -> subject id per window (for LOSO/LOEO)
    """
    blob = smart_load(path)
    X_seq = blob["X_seg"]
    y = blob["y_exercise"]
    groups = blob.get("groups", None)  # Optional but very useful later
    return X_seq, y, groups


def load_test(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load the test dataset.
    Returns:
        X_test : ndarray (M, 50, 45)
        y_test : ndarray (M,) or None (if unlabeled)
    """
    blob = smart_load(path)
    X_test = blob["X_test"]
    y_test = blob.get("y_test_exercise", None)
    return X_test, y_test


def flatten_windows(X_seq: np.ndarray) -> np.ndarray:
    """
    Flatten 3D windows (N, T, F) into 2D (N, T*F) for classical ML models.
    We keep X_seq intact elsewhere for deep learning models.
    """
    return X_seq.reshape(X_seq.shape[0], -1)
