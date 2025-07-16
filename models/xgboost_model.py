from xgboost import XGBClassifier

def get_xgb():
    return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
