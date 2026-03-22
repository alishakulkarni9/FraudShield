import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def run_pipeline(data_path):
    # Get project root (one level above src)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, data_path)

    df = pd.read_csv(full_path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return model, X_test, y_test