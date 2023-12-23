from kazoo.client import KazooClient
from kazoo.recipe.election import Election
import threading
import time
import random
import requests # Importamos la librería requests
import signal # Import the signal module
import os # Import the os module

# Define a function that runs when the interrupt signal is received
def interrupt_handler(signal, frame):
    exit(0)

# Register the function as the interrupt signal handler
signal.signal(signal.SIGINT, interrupt_handler)

# Create an identifier for the application from an env variable
id = os.getenv("APP_ID")

host = "host.docker.internal"
# Create a Kazoo client and connect it to the Zookeeper server
client = KazooClient(hosts=host + ":2181")
client.start()

# Ensure a path, create if necessary
client.ensure_path("/mediciones")

# Create a node with data
try:
    client.create(f"/mediciones/{id}", "a value".encode("utf-8"), ephemeral=True)
except:
    print('Node creation exception (maybe exists)')
    
def leader_func():
    while True:
        print('I am the leader')

        # Get the data of all nodes in the path /mediciones
        children = client.get_children("/mediciones")
        print(f"There are {len(children)} children with names {children}")

        # Filter nodes to get only the ones not containing the "lock" string
        nodes = [child for child in children if "lock" not in child]

        if nodes:
            # Calculate the average of the measurements
            total = 0
            for child in nodes:
                (data, _) = client.get(f"/mediciones/{child}")
                total += int(data.decode("utf-8"))
            average = total / len(nodes)

            # Print the average
            print(f"The average is {average}")

            # Send the average to the server
            url = 'http://' + host + ':4000/nuevo'
            params = {'dato': average}
            response = requests.get(url, params=params)
        else:
            print("No numeric measurements available")

        time.sleep(5)


    
# Define a function that handles the election part
def election_func():
    # Participate in the election with the application identifier
    election.run(leader_func)
    
# Crear una elección entre las aplicaciones y elegir un líder
election = Election(client, "/mediciones" , id)
    
# Create a thread to execute the election_func function
election_thread = threading.Thread(target=election_func, daemon=True)
# Start the thread
election_thread.start()

# Define a function that runs when an application becomes the leader
    
# Periodically generate a new measurement and replace the previous one in the node
while True:
    # Generate a new random measurement
    value = random.randint(10, 40)

    print(f"New measurement: {value}")
    
    # Modify the data of a node
    client.set(f"/mediciones/{id}", str(value).encode("utf-8"))

    # Wait for 5 seconds before the next measurement
    time.sleep(5)
    