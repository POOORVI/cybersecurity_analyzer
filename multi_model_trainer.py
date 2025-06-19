import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import os

# Load log data
LOG_FILE = "cybersecurity_system_logs.csv"
if not os.path.exists(LOG_FILE):
    raise FileNotFoundError(f"Log file '{LOG_FILE}' not found.")

# Read dataset
df = pd.read_csv(LOG_FILE)

# Ensure required column exists
if 'Component' not in df.columns:
    raise ValueError("Column 'Component' is required in the log file for multi-model training.")

# Columns to encode
categorical_columns = ['User_ID', 'IP_Address', 'Location', 'User_Agent', 'Attack_Type']

# Split dataset by Component
log_types = df['Component'].unique()
models = {}

for comp in log_types:
    print(f"\nTraining model for component: {comp}")
    sub_df = df[df['Component'] == comp].copy()

    # Encode categorical features
    for col in categorical_columns:
        if col in sub_df.columns:
            sub_df[col] = LabelEncoder().fit_transform(sub_df[col].astype(str))

    # Select features for training
    feature_cols = ['Session_Duration'] + [col for col in categorical_columns if col in sub_df.columns]
    X = sub_df[feature_cols]

    # Train Isolation Forest
    model = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
    model.fit(X)

    # Add predictions to DataFrame
    sub_df['Predicted_Anomaly'] = model.predict(X)
    sub_df['Anomaly_Score'] = model.decision_function(X)

    # Save model and processed data
    model_filename = f"model_{comp}.joblib"
    output_csv = f"processed_logs_{comp}.csv"
    joblib.dump(model, model_filename)
    sub_df.to_csv(output_csv, index=False)
    print(f"Saved: {model_filename} & {output_csv}")

    models[comp] = model

print("\n✅ All component models trained and saved successfully.")