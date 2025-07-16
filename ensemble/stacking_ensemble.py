from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
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
            ('svm', get_svm())
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )
