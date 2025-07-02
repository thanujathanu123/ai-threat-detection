"""
Create placeholder model files for the AI Threat Detection app.
This script creates empty model files in the app directory.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Create app directory if it doesn't exist
os.makedirs('app', exist_ok=True)

# Create a simple random forest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(np.array([[0, 0, 0, 0, 0, 0, 0]]), np.array([0]))

# Create a simple gradient boosting model
gb_model = GradientBoostingClassifier(n_estimators=10, random_state=42)
gb_model.fit(np.array([[0, 0, 0, 0, 0, 0, 0]]), np.array([0]))

# Save the models
joblib.dump(rf_model, 'app/random_forest_model.pkl')
joblib.dump(gb_model, 'app/gradient_boost_model.pkl')
joblib.dump(rf_model, 'app/model.pkl')  # Use random forest as the default model

print("✅ Created placeholder model files in the app directory:")
print("  - app/random_forest_model.pkl")
print("  - app/gradient_boost_model.pkl")
print("  - app/model.pkl")
print("\nThese are minimal placeholder models for GitHub. For actual use,")
print("train proper models with your network traffic data.")