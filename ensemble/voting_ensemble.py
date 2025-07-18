from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.mlp_model import get_mlp
from models.xgboost_model import get_xgb
from models.svm_model import get_svm

def get_voting_ensemble():
    return VotingClassifier(
        estimators=[
            ('mlp', get_mlp()),
            ('xgb', get_xgb()),
            ('svm', make_pipeline(StandardScaler(), get_svm()))  # SVM needs scaling
        ],
        voting='soft',
        n_jobs=-1  # parallel processing
    )
