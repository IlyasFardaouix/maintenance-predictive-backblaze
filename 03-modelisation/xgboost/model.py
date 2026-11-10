import numpy as np
import xgboost as xgb
from config import THRESHOLD_RUL, BEST_ITERATION, CAP

SMART_COLS = [
    'smart_1_raw','smart_3_raw','smart_4_raw','smart_5_raw',
    'smart_7_raw','smart_9_raw','smart_10_raw','smart_11_raw',
    'smart_12_raw','smart_187_raw','smart_188_raw','smart_189_raw',
    'smart_190_raw','smart_191_raw','smart_192_raw','smart_193_raw',
    'smart_194_raw','smart_195_raw','smart_196_raw','smart_197_raw',
    'smart_198_raw','smart_199_raw','smart_200_raw','smart_240_raw',
]
FEAT_NAMES = [f'{c}_{s}' for c in SMART_COLS
              for s in ['last','mean','min','max','std','slope']]

def summarize_windows(windows: np.ndarray) -> np.ndarray:
    """(N, 30, F) -> (N, F*6) résumés statistiques."""
    t  = np.arange(windows.shape[1], dtype=np.float32)
    tc = t - t.mean()
    sw = (tc / float(np.dot(tc, tc))).astype(np.float32)
    return np.concatenate([
        windows[:, -1, :],
        windows.mean(1), windows.min(1),
        windows.max(1),  windows.std(1),
        np.einsum('ijk,j->ik', windows, sw),
    ], axis=1).astype(np.float32)


class XGBoostModel:
    name      = "xgboost"
    threshold = THRESHOLD_RUL

    def __init__(self, booster, best_iteration=BEST_ITERATION):
        self.booster        = booster
        self.best_iteration = best_iteration

    @classmethod
    def load(cls, folder: str):
        import os
        booster = xgb.Booster()
        booster.load_model(os.path.join(folder, "model.ubj"))
        return cls(booster)

    def predict_rul(self, windows: np.ndarray) -> np.ndarray:
        X  = summarize_windows(windows)
        dm = xgb.DMatrix(X, feature_names=FEAT_NAMES)
        p  = self.booster.predict(
                 dm, iteration_range=(0, self.best_iteration + 1))
        return np.clip(p, 0., CAP).astype(np.float32)

    def alert(self, windows: np.ndarray) -> np.ndarray:
        return self.predict_rul(windows) <= self.threshold