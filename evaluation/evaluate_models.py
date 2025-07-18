# evaluation/evaluate_models.py
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import time
import numpy as np

def evaluate_model(name, model, X_test, y_test, X_train=None, y_train=None):
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    inference_time = end - start

    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Inference Time: {inference_time:.4f} seconds")

    # Optional: Train vs Test score comparison (only if X_train is passed)
    if X_train is not None and y_train is not None:
        train_acc = model.score(X_train, y_train)
        print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {acc:.4f}")
        if train_acc - acc > 0.05:
            print("⚠️ Possible overfitting detected (train accuracy much higher than test).")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

    # Plot ROC Curve
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # Use decision function if no predict_proba
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {name}")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()
    except Exception as e:
        print(f"ROC Curve could not be plotted: {e}")
