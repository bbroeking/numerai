import numpy as np
from sklearn.metrics import r2_score

def our_score(preds, dtrain):
    return "score", -np.corrcoef(preds, dtrain.get_label())[0,1]

def correlation_score(preds, dtrain):
    return "pos_corr", np.corrcoef(preds, dtrain.get_label())[0,1]

def xgb_r2(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(preds, labels)