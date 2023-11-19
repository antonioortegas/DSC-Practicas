import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Cargar los datos del fichero csv "datos.csv"
df = pd.read_csv("practica1/datos.csv", parse_dates=True, index_col=0)

print(df)
print(df.shape)  # (7267,1) (según el fichero cambiará el número de filas)

df.plot()
plt.show()

# Normalizar los datos
scaler = MinMaxScaler()  # Usar esta función
df_scaled = scaler.fit_transform(df)

# Escoger el tamaño de la ventana temporal (n_steps)
n_steps = 10

# Escoger el número de features (n_features) en este caso solo queremos predecir un valor
n_features = 1

# Crear las ventanas temporales
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

# Redimensionar los datos para la RNN
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Crear la RNN
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Cargar el modelo si existe, si no, entrenarlo y guardarlo
model_filename = "modelo"
path = "practica1/" + model_filename + ".keras"
if not os.path.exists(path):
    # Entrenar la RNN
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(X, y, epochs=200,validation_data=(X, y), callbacks=[early_stopping], verbose=1)
    
    # Guardar el modelo
    model.save(path)
else:
    # Cargar el modelo
    model = tf.keras.models.load_model(path)

# Hacemos el predict de todos los datos
y_pred = model.predict(X)

# Calcular el error absoluto medio (MAE)
mae = tf.keras.metrics.mean_absolute_error(y, y_pred).numpy()

# Calcular z-scores y definir anomalías
z_scores = (mae - np.mean(mae)) / np.std(mae)
threshold_z_score = 2.5
anomalies = np.where(z_scores[n_steps-1:] > threshold_z_score)[0] + n_steps - 1

# Crear la columna 'is_anomaly'
df['is_anomaly'] = False
df.loc[df.index.isin(df.index[anomalies]), 'is_anomaly'] = True


# Graficar los datos con anomalías resaltadas
fig, ax = plt.subplots(figsize=(12, 6))

# Line connecting consecutive points
ax.plot(df.index, df['value'], color='blue', label='Temperature', linestyle='-', marker='', markersize=3, linewidth=2)

# Add the predictions to the plot in yellow
y_pred_original_scale = scaler.inverse_transform(y_pred)
ax.plot(df.index[n_steps:], y_pred_original_scale, color='orange', label='Predictions', linestyle='--', marker='', markersize=2, linewidth=0.75)

# Highlight anomalies
ax.scatter(df.index[df['is_anomaly']], df['value'][df['is_anomaly']], color='red', label='Anomalies', s=50, zorder=5)

# Additional styling
ax.legend()
ax.set_xlabel('Timestamp')
ax.set_ylabel('Temperature')
ax.set_title('Temperature with Anomalies Highlighted')
ax.grid(True)
plt.show()