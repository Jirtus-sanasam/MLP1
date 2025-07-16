from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time

def evaluate_model(name, model, X_test, y_test):
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
    print(f"Inference Time: {end - start:.4f} seconds")
