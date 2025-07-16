from sklearn.ensemble import VotingClassifier
from models.mlp_model import get_mlp
from models.xgboost_model import get_xgb
from models.svm_model import get_svm

def get_voting_ensemble():
    return VotingClassifier(
        estimators=[
            ('mlp', get_mlp()),
            ('xgb', get_xgb()),
            ('svm', get_svm())
        ],
        voting='soft'
    )
