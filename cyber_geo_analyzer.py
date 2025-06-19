import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import os
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score
import geoip2.database  # Import the geoip2 library for GeoIP lookups

warnings.filterwarnings('ignore')

class CybersecurityLogAnalyzer:
    def __init__(self, csv_file_path='cybersecurity_system_logs.csv'):
        self.csv_file_path = csv_file_path
        self.df = None
        self.model = None
        self.label_encoders = {}
        self.categorical_columns = ['User_ID', 'IP_Address', 'Location', 'User_Agent', 'Attack_Type']

    def load_data(self):
        if not os.path.exists(self.csv_file_path):
            print(f"Error: File '{self.csv_file_path}' not found!")
            return False
        self.df = pd.read_csv(self.csv_file_path)
        return True

    def preprocess_data(self):
        if self.df is None:
            print("Data not loaded.")
            return False
        self.df_processed = self.df.copy()

        for column in self.df_processed.columns:
            if self.df_processed[column].dtype == 'object':
                self.df_processed[column] = self.df_processed[column].fillna('Unknown')
            else:
                self.df_processed[column] = self.df_processed[column].fillna(0)

        try:
            self.df_processed['Timestamp'] = pd.to_datetime(self.df_processed['Timestamp'])
            self.df_processed['Hour'] = self.df_processed['Timestamp'].dt.hour
            self.df_processed['DayOfWeek'] = self.df_processed['Timestamp'].dt.dayofweek
            self.df_processed['IsWeekend'] = self.df_processed['DayOfWeek'].isin([5, 6]).astype(int)
            self.df_processed['IsNight'] = self.df_processed['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
        except:
            self.df_processed['Hour'] = 12
            self.df_processed['DayOfWeek'] = 1
            self.df_processed['IsWeekend'] = 0
            self.df_processed['IsNight'] = 0

        for column in self.categorical_columns:
            if column in self.df_processed.columns:
                le = LabelEncoder()
                self.df_processed[f'{column}_encoded'] = le.fit_transform(self.df_processed[column].astype(str))
                self.label_encoders[column] = le

        feature_columns = ['Hour', 'DayOfWeek', 'IsWeekend', 'IsNight']
        if 'Session_Duration' in self.df_processed.columns:
            feature_columns.append('Session_Duration')
        encoded_columns = [f'{col}_encoded' for col in self.categorical_columns if f'{col}_encoded' in self.df_processed.columns]
        feature_columns.extend(encoded_columns)

        self.X = self.df_processed[feature_columns].fillna(0)
        return True

    def train_isolation_forest(self, contamination=0.10):
        if self.X is None:
            print("Preprocessing required.")
            return False
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples='auto',
            max_features=1.0
        )
        self.model.fit(self.X)
        self.df_processed['Predicted_Anomaly'] = self.model.predict(self.X)
        self.df_processed['Anomaly_Score'] = self.model.decision_function(self.X)
        return True

    def enrich_with_geoip(self, db_path='GeoLite2-City.mmdb'):
        import geoip2.database

        if self.df is None or 'IP_Address' not in self.df.columns:
            print("Missing data or IP_Address column.")
            return False

        self.df['GeoCity'] = 'Unknown'
        self.df['GeoCountry'] = 'Unknown'

        try:
            reader = geoip2.database.Reader(db_path)
        except Exception as e:
            print("Could not load GeoLite2 DB:", e)
            return False

        for idx, ip in self.df['IP_Address'].items():
            try:
                response = reader.city(ip)
                if response:
                    self.df.at[idx, 'GeoCity'] = response.city.name or 'Unknown'
                    self.df.at[idx, 'GeoCountry'] = response.country.iso_code or 'Unknown'
            except:
                pass

        reader.close()
        return True

    def evaluate_model(self):
        if 'Is_Anomalous' in self.df_processed.columns:
            y_true = self.df_processed['Is_Anomalous']
            y_pred = self.df_processed['Predicted_Anomaly'].apply(lambda x: 1 if x == -1 else 0)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            print(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")

    def save_model_and_data(self, model_path='isolation_forest_model.joblib', data_path='processed_cybersecurity_logs.csv'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoders, 'label_encoders.joblib')
        self.df_processed.to_csv(data_path, index=False)
        return True

    def plot_anomaly_distribution(self, in_streamlit=False):
        plt.figure(figsize=(10, 4))
        plt.hist(self.df_processed['Anomaly_Score'], bins=100, color='teal')
        plt.title("Anomaly Score Distribution")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        if in_streamlit:
            st.pyplot(plt.gcf())
        else:
            plt.show()

    def generate_summary_report(self):
        if self.df_processed is None:
            return
        total_logs = len(self.df_processed)
        anomalies = len(self.df_processed[self.df_processed['Predicted_Anomaly'] == -1])
        print(f"Total Logs: {total_logs}\nAnomalies: {anomalies} ({anomalies/total_logs*100:.2f}%)")


def main():
    analyzer = CybersecurityLogAnalyzer()
    if not analyzer.load_data():
        return
    if not analyzer.preprocess_data():
        return
    if not analyzer.train_isolation_forest():
        return
    analyzer.evaluate_model()
    analyzer.save_model_and_data()
    analyzer.plot_anomaly_distribution(in_streamlit=False)
    analyzer.generate_summary_report()

if __name__ == "__main__":
    main()