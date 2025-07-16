import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.mlp_model import get_mlp
from models.xgboost_model import get_xgb
from models.svm_model import get_svm
from models.rf_model import get_rf
from ensemble.voting_ensemble import get_voting_ensemble
from ensemble.stacking_ensemble import get_stacking_ensemble
from evaluation.evaluate_models import evaluate_model


def load_and_preprocess_data(path):
    """Load and preprocess the diabetes dataset."""
    print("🔄 Loading and preprocessing data...")
    try:
        data = pd.read_csv(path)
        X = data.drop("Outcome", axis=1)
        y = data["Outcome"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("✅ Data loaded and scaled.")
        return X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit(1)


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/diabetes.csv")

    # Define all models
    models = {
        "MLP": get_mlp(),
        "XGBoost": get_xgb(),
        "SVM": get_svm(),
        "Random Forest": get_rf(),
        "Voting Ensemble": get_voting_ensemble(),
        "Stacking Ensemble": get_stacking_ensemble()
    }

    print("\n🚀 Training and evaluating models...")
    for name, model in models.items():
        try:
            print(f"\n🔧 Training: {name}")
            model.fit(X_train, y_train)
            evaluate_model(name, model, X_test, y_test)
        except Exception as e:
            print(f"❌ Error in model {name}: {e}")

    print("\n🏁 All models evaluated.")


if __name__ == "__main__":
    main()
