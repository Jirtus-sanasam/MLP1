from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.mlp_model import get_mlp
from models.xgboost_model import get_xgb
from models.rf_model import get_rf
from models.svm_model import get_svm

def get_stacking_ensemble():
    return StackingClassifier(
        estimators=[
            ('mlp', get_mlp()),
            ('xgb', get_xgb()),
            ('rf', get_rf()),
            ('svm', make_pipeline(StandardScaler(), get_svm()))  # SVM benefits from scaling
        ],
        final_estimator=LogisticRegression(max_iter=1000, solver='lbfgs'),
        passthrough=False,
        cv=5,
        n_jobs=-1
    )
