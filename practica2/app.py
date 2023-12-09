import os
import json
import random
import socket
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from redis import Redis, RedisError
from flask import Flask, request, render_template, jsonify
from datetime import datetime
from subprocess import run
from keras.models import load_model

# Connect to Redis
REDIS_HOST = os.getenv('REDIS_HOST', "localhost")
redis = Redis(host=REDIS_HOST, db=0, socket_connect_timeout=2, socket_timeout=2)

# CONFIGURATION
series_name = "temperature" # Name of the time series in Redis
model = load_model('models/lstm.h5') # Load the LSTM model
scaler = joblib.load('models/scaler.pkl') # Load the scaler

# Function to read values from config.txt
def read_config(file_path="models/config.txt"):
    config_values = {}
    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split(" = ")
            # Convert values to appropriate types if needed
            if key in ["n_steps", "threshold_z_score"]:
                config_values[key] = int(value)
            else:
                config_values[key] = float(value)
    return config_values
# Read values from config.txt
# This values were obtained from the model in the previous step, (/practica1/lstm.py)
config_data = read_config()

# Access variables, store 0 if not found
n_steps = config_data.get("n_steps", 0)
threshold_z_score = config_data.get("threshold_z_score", 0)
std = config_data.get("std", 0.0)
mean = config_data.get("mean", 0.0)

# Now you can use these variables in your application
print(f"n_steps: {n_steps}, threshold_z_score: {threshold_z_score}, std: {std}, mean: {mean}")

# Create the time series in Redis if it doesn't exist
try:
    redis.execute_command('TS.CREATE', series_name)
except RedisError:
    print("Series already exists")

# Function to add 100 random values to Redis
def add_one_hundred_values():
    # Run the script to add 100 random values
    # Create a TS called 'temperature' if it doesn't exist 
    try:
        redis.execute_command('TS.CREATE', 'temperature')
    except Exception as e:
        print(e)

    # Get the current time before the loop
    current_time = int(time.time() * 1000)  # Convert to milliseconds

    # Run the loop 100 times
    for i in range(100):
        temp = random.randint(10, 40)
        
        # Each iteration, decrement the current time by 3 seconds to avoid overwriting values
        current_time -= 3000
        
        redis.execute_command('TS.ADD', 'temperature', str(current_time), temp)
        print(f'Temperature: {temp} °C at timestamp: {current_time}')

    
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


# Create the Flask application
app = Flask(__name__, static_url_path='/static')

# Define the main route
@app.route("/")
def homepage():
    return render_template("homepage.html", hostname=socket.gethostname())

# Add a new value to Redis
@app.route("/nuevo")
def nuevo():
    try:
        dato = request.args.get('dato') # Get the new measurement from the query parameters
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]  # Readable timestamp
        key = f"{timestamp}"
        redis.set(key, dato)  # Store each measurement with a unique key
        redis.execute_command('TS.ADD', series_name, '*', dato)
        print(f"Added {dato} to {series_name}")
    except RedisError:
        dato = "<i>Redis error</i>"

    html = "<b>Value:</b> {dato} -> correctly saved with timestamp: {key}"
    return html.format(dato=dato, key=key)

# List all values in Redis
@app.route("/listar")
def listar():
    try:
        all_values = redis.execute_command('TS.RANGE', series_name, '-', '+') # Get all values from the series "temperature"
        measurements = {} # Initialize an empty dictionary to store key-value pairs
        # Iterate through all values and store them in the dictionary
        for value in all_values:
            timestamp = value[0]
            dato = value[1]
            # Convert dato from b"value" to value
            dato = dato.decode()
            timestamp = timestamp / 1000
            dt = datetime.fromtimestamp(timestamp)
            dt_str = dt.strftime('%d/%m/%Y %H:%M:%S')
            measurements[dt_str] = dato
        # Invert the measurements dictionary to have the latest values first
        measurements = dict(reversed(list(measurements.items())))

    except RedisError:
        measurements = {"error": "cannot connect to Redis"}
        
    # Display measurements as a list, each measurement in a new line
    html = "<h3>Values in Redis:</h3>"
    for key, value in measurements.items():
        html += f"<b>{key} : </b>{value}<br/>"
    
    return html

# Detect if a given value is an anomaly
@app.route("/detectar")
def detectar():
    try:
        # Get the new measurement from the query parameters
        nuevo_dato = float(request.args.get('dato'))

        # Store the new measurement in Redis
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
        key = f"measurement : {timestamp}"
        redis.set(key, nuevo_dato)
        redis.execute_command('TS.ADD', series_name, '*', nuevo_dato)
        print(f"Added {nuevo_dato} to {series_name}")

        # Get the latest [n_steps] values from Redis
        muestras = redis.execute_command('TS.REVRANGE', series_name, '-', '+', 'COUNT', n_steps)

        # Extract values and timestamps
        valores = [m[1] for m in muestras]
        timestamps = [m[0] / 1000 for m in muestras]

        # Convert timestamps to datetime objects
        fechas = [datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S') for ts in timestamps]

        # Prepare the input sequence for the LSTM model
        input_sequence = np.array(valores[::-1])  # Reverse the order to match the model's expectations
        # Add the new measurement to the input sequence
        input_sequence = np.append(input_sequence, nuevo_dato)

        # Create a dataframe with the input sequence
        df = pd.DataFrame(input_sequence)
        df_scaled = scaler.fit_transform(df)

        X, y = split_sequence(df_scaled, n_steps)
        
        y_pred = model.predict(X)

        # Calculate z-score for anomalies detection
        mae = tf.keras.metrics.mean_absolute_error(y, y_pred).numpy()
        z_score = (mae - mean) / std

        # Determine if there is an anomaly based on the threshold_z_score
        anomaly_detected = z_score > threshold_z_score

        # Create a response JSON
        response = {
            "mediciones": [{"time": fecha, "valor": valor} for fecha, valor in zip(fechas, valores)],
            "anomalia": "YES" if anomaly_detected else "NO"
        }

    # Handle exceptions
    except RedisError:
        response = {"error": "cannot connect to Redis"}
    except Exception as e:
        response = {"error": str(e)}

    # Build and return the response
    response_json = json.loads(json.dumps(response, default=str))

    return jsonify(response_json)

# Reset the database
@app.route("/reset")
def reset():
    try:
        redis.flushall()
        redis.execute_command('TS.CREATE', series_name)
    except RedisError:
        return "Error"
    return "Database has been correctly reset."

# Add 100 random values to Redis
@app.route("/add_one_hundred_values")
def add_one_hundred_values_route():
    try:
        add_one_hundred_values()
        result = "Successfully added 100 random values."
    except Exception as e:
        result = f"Error: {str(e)}"

    return jsonify({"result": result})

# Run the application
# This is the entry point for the Docker container
# The port can be configured using an environment variable, otherwise it will default to 80
if __name__ == "__main__":
    PORT = os.getenv('PORT', 80)
    app.run(host='0.0.0.0', port=PORT, debug=False)
