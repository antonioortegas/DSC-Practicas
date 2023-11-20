import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Load the data from the CSV file "datos.csv"
df = pd.read_csv("practica1/datos.csv", parse_dates=True, index_col=0)

print(df.shape)  # (7267,1) in this case

# Show the original data
df.plot()
plt.show()

# Normalize the data so that all values are between 0 and 1
scaler = MinMaxScaler()  # Create the scaler object
df_scaled = scaler.fit_transform(df)

# Choose the number of time steps (n_steps) and the number of features (n_features)
n_steps = 25
n_features = 1

# Create the time series sequences
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X, y = split_sequence(df_scaled, n_steps)

# Reshape the data for the LSTM
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Create the LSTM model
# I am using two LSTM layers with 50 neurons each, and a Dense layer with 1 neuron as the output layer.
# Model seemed to be overfitting after some tests, so I added two Dropout layers to try to reduce it
# Although I did not try many different configurations, at 0.2 dropout rate the model exhibited more similarities to the provided example
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(Dropout(0.4))  # Adding dropout for regularization
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.4))  # Adding dropout for regularization
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# Check if the model has already been trained and saved to disk. If not, train it and save it.
# This is done to avoid training the model every time the script is run.
# If you want to retrain the model, delete the file "LSTM.keras"
model_filename = "lstm"
path = "practica1/" + model_filename + ".keras"
if os.path.exists(path):
    model = tf.keras.models.load_model(path)
else:
    # Here I am using an EarlyStopping callback to stop the training when the model stops improving for a defined number of epochs (patience).
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X, y, epochs=100,validation_data=(X, y), callbacks=[early_stopping], verbose=1)
    model.save(path) # Save the model to disk
     
# Predict the values with the model
y_pred = model.predict(X)

# Calculate the mean absolute error (MAE)
mae = tf.keras.metrics.mean_absolute_error(y, y_pred).numpy()

# Calculate the z-scores
z_scores = (mae - np.mean(mae)) / np.std(mae)
threshold_z_score = 3.0

# Detect the anomalies
anomalies = np.where(z_scores[n_steps-1:] > threshold_z_score)[0] + n_steps

# Create a new column with the predictions
df['is_anomaly'] = False
df.loc[df.index.isin(df.index[anomalies]), 'is_anomaly'] = True

# Print the anomalies detected (their dates) as a single column
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
