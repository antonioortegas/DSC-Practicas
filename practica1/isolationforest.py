import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the data from the CSV file "datos.csv"
df = pd.read_csv("practica1/datos.csv", parse_dates=True, index_col=0)

# Normalize the data so that all values are between 0 and 1
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Choose the number of time steps (n_steps) and the number of features (n_features)
n_steps = 25
n_features = 1

# Create the time series sequences
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X, y = split_sequence(df_scaled, n_steps)

# Reshape the data for Isolation Forest
X = X.reshape((X.shape[0], X.shape[1] * n_features))

# Train the Isolation Forest model
model_filename = "practica1/isolationforest.joblib"
if os.path.exists(model_filename):
    clf = joblib.load(model_filename)
else:
    clf = IsolationForest(verbose=1, contamination=0.03) # I have tried different values for contamination, and 0.02 seems to be the most appropriate
    clf.fit(X)
    joblib.dump(clf, model_filename)

# Make predictions with the trained model
y_pred = clf.predict(X)

# Identify anomalies (outliers)
anomalies = np.where(y_pred == -1)[0]

# Update the DataFrame with anomaly information
df['is_anomaly_iforest'] = False
df.loc[df.index.isin(df.index[anomalies]), 'is_anomaly_iforest'] = True

# Print the anomalies detected (their dates) as a single column
print("El número de anomalías detectadas con Isolation Forest es: ", len(anomalies))
anomaly_list_iforest = df[df['is_anomaly_iforest']].index.to_list()
for i in anomaly_list_iforest:
    print(i)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['value'], color='blue', label='Temperature', linestyle='-', marker='', markersize=3, linewidth=2)
ax.scatter(df.index[df['is_anomaly_iforest']], df['value'][df['is_anomaly_iforest']], color='red', label='Anomalies (Isolation Forest)', s=50, zorder=5)
ax.legend()
ax.set_xlabel('Timestamp')
ax.set_ylabel('Temperature')
ax.set_title('Temperature with Isolation Forest')
ax.grid(True)
plt.show()
