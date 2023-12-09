# Importar los módulos necesarios
import redis
import random
import time

# Crear una conexión a redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Crear una serie temporal llamada 'temperature'
try:
    r.execute_command('TS.CREATE', 'temperature')
except Exception as e:
    print(e)

# Capturar el tiempo actual antes del bucle
current_time = int(time.time() * 1000)  # Convert to milliseconds

# Misma funcionalidad pero el bucle se ejecuta 100 veces
for i in range(100):
    temp = random.randint(10, 40)
    
    # Decrementar el tiempo actual por 3 segundos por iteración para no sobreescribir los valores
    current_time -= 3000
    
    r.execute_command('TS.ADD', 'temperature', str(current_time), temp)
    print(f'Temperature: {temp} °C at timestamp: {current_time}')
