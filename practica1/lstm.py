# Import necessary libraries
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Load time series data from the CSV file "datos.csv"
df = pd.read_csv("practica1/datos.csv", parse_dates=True, index_col=0)

# Display the shape of the loaded data
print(df.shape)  # (7267, 1) in this case

# Visualize the original data
df.plot()
plt.show()

# Normalize the data to have values between 0 and 1
scaler = MinMaxScaler()  # Create MinMaxScaler object
df_scaled = scaler.fit_transform(df)

# Define parameters for time series sequences
n_steps = 10
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
print(X)

# Reshape the data for input to LSTM model
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Create an LSTM model with dropout for regularization
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(Dropout(0.4))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Check if the model has been trained and saved to disk; if not, train and save it
model_filename = "lstm"
path = "practica1/" + model_filename + ".h5"
if os.path.exists(path):
    model = tf.keras.models.load_model(path)
else:
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X, y, epochs=100, validation_data=(X, y), callbacks=[early_stopping], verbose=1)
    model.save(path)
    joblib.dump(scaler, "practica1/scaler.pkl")

# Predict values using the trained model
y_pred = model.predict(X)
print(y_pred)

# Calculate mean absolute error (MAE)
mae = tf.keras.metrics.mean_absolute_error(y, y_pred).numpy()

# Calculate z-scores for anomalies detection
z_scores = (mae - np.mean(mae)) / np.std(mae)
threshold_z_score = 3.0
print("Z_SCORES" + "\n")
print(z_scores.shape)
print(mae.shape)
print(np.mean(mae))
print(np.std(mae))

# Detect anomalies based on z-scores
anomalies = np.where(z_scores[n_steps - 1:] > threshold_z_score)[0] + n_steps

# Create a new column indicating anomalies in the original dataframe
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
y_pred_original_scale = scaler.inverse_transform(y_pred)
ax.plot(df.index[n_steps:], y_pred_original_scale, color='orange', label='Predictions', linestyle='--', marker='', markersize=2, linewidth=0.75)
ax.scatter(df.index[df['is_anomaly']], df['value'][df['is_anomaly']], color='red', label='Anomalies', s=50, zorder=5)
ax.legend()
ax.set_xlabel('Timestamp')
ax.set_ylabel('Temperature')
ax.set_title('Temperature with LSTM')
ax.grid(True)
plt.show()