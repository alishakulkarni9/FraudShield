import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.sample(20000, random_state=42)  # use only 20k rows

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler