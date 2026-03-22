from sklearn.metrics import classification_report, roc_auc_score
import json
import os

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    report["roc_auc"] = roc_auc_score(y_test, y_probs)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/metrics.json", "w") as f:
        json.dump(report, f, indent=4)