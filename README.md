This project implements a machine learning system for detecting anomalous patterns in cybersecurity attack data. The system combines Isolation Forest and Autoencoder techniques to identify potential security threats in network traffic data.

Key Features

1. Data Preprocessing: Handles missing values, categorical encoding, and timestamp feature extraction
2. Dual Detection Approach:
  a. Isolation Forest: Unsupervised algorithm for detecting outliers
  b. Autoencoder: Deep learning model that learns to reconstruct normal patterns
3. API Integration: Flask-based REST API for real-time anomaly detection
4. Visualization: Includes data visualization of detection results

Technical Stack

- Python (Pandas, NumPy, Scikit-learn)
- TensorFlow/Keras for deep learning
- Flask for API deployment
- Matplotlib/Seaborn/Plotly for visualization

Dataset

The system analyzes the "cybersecurity_attacks.csv" dataset containing:
- Network traffic features
- Security alerts/warnings
- Malware indicators
- Firewall/IDS/IPS logs
- Attack type classifications

Usage

1. Train the models on historical data
2. Use the Flask API (`/detect` endpoint) to check new records for anomalies
3. Visualize detection patterns through generated plots

Applications

- Network intrusion detection
- Threat hunting
- Security monitoring systems
- SIEM (Security Information and Event Management) enhancement

Output

![image](https://github.com/user-attachments/assets/d2f617df-54db-48ec-a4c0-9b7432f68c27)
![image](https://github.com/user-attachments/assets/53008e51-21ec-42d5-8a5e-3df059833608)

