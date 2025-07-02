# data/predict.py

import pandas as pd
import joblib
import os
from data.data_preprocessing import preprocess_data

def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "model.pkl")
    return joblib.load(model_path)

def predict_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    X, _ = preprocess_data(df)
    model = load_model()
    predictions = model.predict(X)
    return predictions
