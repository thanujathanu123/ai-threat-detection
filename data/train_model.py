# data/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from data.data_preprocessing import preprocess_data

def train_and_save_model():
    df = pd.read_csv("data/raw/your_dataset.csv")  # Replace with actual CSV
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

if __name__ == "__main__":
    train_and_save_model()
