from xgboost import XGBClassifier

def get_xgb():
    return XGBClassifier(
        n_estimators=200,           # more trees for better averaging
        max_depth=4,                # shallower trees to reduce overfitting
        learning_rate=0.05,         # slower learning improves generalization
        subsample=0.8,              # row sampling
        colsample_bytree=0.8,       # feature sampling
        gamma=1,                    # minimum loss reduction to make a split
        reg_alpha=0.5,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
