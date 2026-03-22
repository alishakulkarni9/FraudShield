from sklearn.model_selection import train_test_split
import joblib
from imblearn.over_sampling import SMOTE

from src.data import load_and_preprocess
from src.model import train_model
from src.evaluate import evaluate

# Load data
X, y, scaler = load_and_preprocess("data/creditcard.csv")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train
model = train_model(X_train, y_train)

# Save
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Evaluate
evaluate(model, X_test, y_test)

print("✅ Pipeline completed successfully!")