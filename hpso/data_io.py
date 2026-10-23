import os
import numpy as np
import pandas as pd


def load_section(path):
    """text matrix (n_samples, n_traces) or .npy"""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    return np.loadtxt(path).astype(np.float64)


def load_well(path, n_samples):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    return df.iloc[:n_samples].reset_index(drop=True)


def get_paths(cfg):
    return os.path.join(cfg.DATA_DIR, cfg.SEISMIC_FILE), os.path.join(cfg.DATA_DIR, cfg.WELL_FILE)
