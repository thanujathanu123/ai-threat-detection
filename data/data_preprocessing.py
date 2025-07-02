# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    # Drop non-useful columns (example: IPs, ports)
    df = df.drop(columns=['Flow ID', 'Source IP', 'Destination IP'], errors='ignore')

    # Fill missing values
    df.fillna(0, inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Separate features and labels
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
