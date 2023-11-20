# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load time series data from the CSV file "datos.csv"
df = pd.read_csv("practica1/datos.csv", parse_dates=True, index_col=0)

# Normalize the data to have values between 0 and 1
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Define parameters for time series sequences
n_steps = 25
n_features = 1

# Create time series sequences using a sliding window approach
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

# Reshape the data for input to the LSTM autoencoder
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Define the LSTM autoencoder model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dropout(0.2))  # Added dropout layer
model.add(RepeatVector(n_steps))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))  # Added dropout layer
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

# Check if the model has been trained and saved to disk; if not, train and save it
model_filename = "autoencoder"
path = "practica1/" + model_filename + ".keras"
if os.path.exists(path):
    model = tf.keras.models.load_model(path)
else:
    model.fit(X, X, epochs=100, verbose=1)
    model.save(path)

# Predict values with the LSTM autoencoder
yhat = model.predict(X, verbose=0)

# Calculate the reconstruction error (mean squared error)
reconstruction_error = np.mean(np.square(X - yhat), axis=(1, 2))

# Calculate z-scores for anomalies detection
z_scores = (reconstruction_error - np.mean(reconstruction_error)) / np.std(reconstruction_error)
threshold_z_score = 2.5

# Detect anomalies based on z-scores
anomalies = np.where(z_scores > threshold_z_score)[0]

# Update the DataFrame with anomaly information
df['is_anomaly'] = False
df.loc[df.index.isin(df.index[anomalies]), 'is_anomaly'] = True

# Print the detected anomalies (dates) and the number of anomalies
print("El número de anomalías detectadas es: ", len(anomalies))
anomaly_list = df[df['is_anomaly']].index.to_list()
for i in anomaly_list:
    print(i)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['value'], color='blue', label='Temperature', linestyle='-', marker='', markersize=3, linewidth=2)
ax.scatter(df.index[df['is_anomaly']], df['value'][df['is_anomaly']], color='red', label='Anomalies', s=50, zorder=5)
ax.legend()
ax.set_xlabel('Timestamp')
ax.set_ylabel('Temperature')
ax.set_title('Temperature with Autoencoder')
ax.grid(True)
plt.show()
