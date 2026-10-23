import numpy as np


def scores(true, pred):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    cc = np.corrcoef(true, pred)[0, 1]
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return {
        "CC": float(cc),
        "R2": float(1 - ss_res / ss_tot),
        "RMSE": float(np.sqrt(np.mean((true - pred) ** 2))),
        "MAE": float(np.mean(np.abs(true - pred))),
        "MAPE_%": float(np.mean(np.abs((true - pred) / true)) * 100),
        "NRMSE": float(np.sqrt(np.mean((true - pred) ** 2)) / (true.max() - true.min())),
    }
