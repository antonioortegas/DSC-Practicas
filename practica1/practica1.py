# Muestra los valores más anómalos del fichero csv suministrado

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# Cargar los datos
df = pd.read_csv("datos.csv",index_col=0,parse_dates=True)

print(df)
print(df.shape) # (7267,1) (según el fichero cambiará el número de filas)

df.plot()
plt.show()

# Normalizar los datos
scaler = MinMaxScaler() # Usar esta función
df_scaled = scaler.fit_transform(df)
print(df_scaled)

# Crear las ventanas temporales
# Lo que se predice (y) es el "siguiente" valor de la secuencia (sin normalizar)
# pasando la ventana actual que tenemos.

# Dividir los datos en entrenamiento y prueba es lo habitual
# Aunque en este caso, vamos a querer luego detectar anomalías en todos los datos

# Redimensionar los datos para la RNN
# LSTM espera 3 dimensiones: número muestras, pasos temporales, número features
# P.ej: (5805,10,1)

# Crear la RNN

# Entrenar la RNN

# Un posible criterio de anomalía
#  Calcular el error absoluto medio (MAE)
# mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()

# Otro posible criterio: uso de percentiles

# Mostrar las fechas de las anomalías

# los valores de anomalias se refieren a las ventanas, no a valores específicos
