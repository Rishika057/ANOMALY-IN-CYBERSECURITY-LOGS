import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify
import threading
import json
app = Flask(__name__)
df = pd.read_csv("cybersecurity_attacks.csv", on_bad_lines='skip', quoting=3)
df.head()
df.fillna("Unknown", inplace=True)
le = LabelEncoder()
for col in ['Alerts/Warnings', 'Malware Indicators', 'Proxy Information', 'Firewall Logs', 'IDS/IPS Alerts', 'Traffic Type', 'Attack Type']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df.drop(columns=['Timestamp'], inplace=True, errors='ignore')
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['Anomaly_IsolationForest'] = iso_forest.fit_predict(df_scaled)
input_dim = df_scaled.shape[1]
autoencoder = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(input_dim,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(df_scaled, df_scaled, epochs=20, batch_size=32, verbose=0)
reconstructions = autoencoder.predict(df_scaled)
mse = np.mean(np.power(df_scaled - reconstructions, 2), axis=1)
df['Anomaly_Autoencoder'] = mse > np.percentile(mse, 95)
df['Final_Anomaly'] = (df['Anomaly_IsolationForest'] == -1) | (df['Anomaly_Autoencoder'] == True)
@app.route('/detect', methods=['POST'])
def detect_anomaly():
    record = request.json
    record_df = pd.DataFrame([record])
    record_df.fillna("Unknown", inplace=True)
    for col in ['Alerts/Warnings', 'Malware Indicators', 'Proxy Information', 'Firewall Logs', 'IDS/IPS Alerts', 'Traffic Type', 'Attack Type']:
        if col in record_df.columns:
            record_df[col] = le.transform([record_df[col][0]])[0]
    record_df = record_df.select_dtypes(include=[np.number])  # Ensure only numerical data
    record_df_scaled = scaler.transform(record_df)
    anomaly_forest = iso_forest.predict(record_df_scaled)[0]
    reconstruction = autoencoder.predict(record_df_scaled)
    anomaly_autoencoder = np.mean(np.power(record_df_scaled - reconstruction, 2), axis=1) > np.percentile(mse, 95)
    final_anomaly = (anomaly_forest == -1) or (anomaly_autoencoder[0])
    return jsonify({"Anomaly": final_anomaly, "Record": record})
def run_flask():
    app.run(host='0.0.0.0', port=5000)
threading.Thread(target=run_flask, daemon=True).start()

plt.figure(figsize=(10, 5))
sns.countplot(x=df['Final_Anomaly'])
plt.title("Anomaly Detection Results")
plt.xlabel("Anomaly Detected")
plt.ylabel("Count")
plt.show()

