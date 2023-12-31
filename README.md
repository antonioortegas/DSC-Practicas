# DSC-Prácticas

A continuación, se describen las instrucciones para ejecutar cada una de las prácticas.

## Práctica 1

El propósito principal de esta práctica es desarrollar y aplicar modelos de Machine Learning para detectar posibles anomalías en una serie temporal de temperaturas. Se explorarán tres enfoques: LSTM, autoencoders y Isolation Forest. Cada uno de estos métodos ofrece distintas perspectivas y técnicas para identificar patrones inusuales en los datos.

**Instrucciones para ejecutar el código:**

- Se necesita tener instalado Python (en mi caso ha utilizado 3.11.3)
- Clonar el repositorio o descargar los archivos en una carpeta local.
- Abrir una terminal y situarse en la raíz del repositorio
- Ejecutar el comando `pip install -r practica1/requirements.txt` para instalar las librerías necesarias (esto se puede hacer en un entorno virtual)
- Ejecutar cualquiera de los archivos ".py" correspondientes a cada uno de los modelos (LSTM, autoencoders o Isolation Forest)

## Práctica 2

El propósito principal de esta práctica es una introducción a Docker (Comandos básicos, creación de contenedores, fichero compose, swarms...). Crearemos una imagen de Docker con un modelo de Machine Learning y la desplegaremos en un contenedor. El modelo de Machine Learning será el mismo que el de la práctica 1.
La idea es que el modelo de Machine Learning se ejecute en un contenedor de Docker, y que se pueda acceder a él a través de una API REST. Podremos consultar los datos almacenados, introducir nuevos datos y obtener predicciones.
La imagen de Docker se creará a partir de un Dockerfile, desde la imagen base de Python.
[DockerHub](https://hub.docker.com/repository/docker/aortegasdev/practica2)
Para el desarrolo de nuestra aplicación, utilizaremos Flask, que es un framework de Python para crear aplicaciones web.
Durante el proceso de desarrollo, se ha utilizado un contenedor de Redis preexistente, preparado para tratar con datos de series temporales, por lo que nos hemos centrado en el desarrollo de la aplicación web y en la conexión con el contenedor de Redis.
También usamos Grafana para visualizar los datos de Redis.
Por último, se ha utilizado un visualizador de contenedores para comprobar que todo funciona correctamente.

**Instrucciones para ejecutar el código:**

- Se necesita **tener Docker Desktop instalado y Docker Engine en funcinamiento** en caso de querer ejecutar los contenedores.
- Se necesita **tener instalado Python** (en mi caso ha utilizado 3.11.3) en caso de querer ejecutar la aplicación de forma local desde el archivo python.
- **Clonar el repositorio** o descargar los archivos en una carpeta local.
- **Abrir una terminal y situarse en la raíz del repositorio**.
- Ejecutar el comando para instalar las librerías necesarias (esto se puede hacer en un entorno virtual)

```terminal
pip install -r practica2/requirements.txt
```

- Una vez llegados a este punto, para probar la aplicación basta con añadir nuestra máquina como swarm, y ejecutar el comando para desplegar el stack de Docker (esta operación puede tardar, ya que deberá descargar todas las imágenes y construir los contenedores).

```text
docker swarm init
docker stack deploy -c practica2/docker-compose.yml NOMBRE
```

> [!TIP]
> Podemos situarnos en la carpeta practica2 y ejecutar el comando `docker-compose pull` para descargar las imágenes necesarias por separado antes de ejecutar el stack.

- Accedemos a la aplicación a través de

```text
localhost:4000
```

Para parar el stack de Docker, podemos hacer `docker stack rm NOMBRE`.

[A continuación se describen las diferentes opciones para ejecutar y compilar la aplicación de forma local.]

- Para ejecutar nuestra practica tenemos cuatro opciones (la primera opción es la recomendable durante el desarrollo mientras estamos haciendo cambios, el resto son para comprobar que todo funciona correctamente):
  - *1.- Ejecutar nuestra aplicación como un archivo python y conectarla a Redis.*
    - Ejecutar el comando `python practica2/app.py` (Esto ejecutará la aplicación en localhost:80). La aplicación no estaría conectada a Redis, por lo que no se podrían hacer consultas ni predicciones.
    - Para usar Redis con la aplicación, en este caso podemos hacer `docker run --name some-redis -p 6379:6379 -d redislabs/redistimeseries`. Esto ejecutará un contenedor de Redis en el puerto 6379. Ahora nuestra aplicación está comunicada con Redis. Para parar el contenedor de Redis, podemos hacer `docker stop some-redis` y para volver a iniciarlo `docker start some-redis`.
  - *2.- Ejecutar nuestra aplicación como un contenedor de Docker y conectarla a Redis.*
    - En caso de no querer crear la imagen localmente desde el Dockerfile, podemos ejecutar el comando `docker run -p 4000:80 aortegasdev/practica2` para descargar la imagen directamente de Docker Hub y ejecutarla.
    - Otra opción es ejecutar el comando `docker build -t NOMBRE ./practica2` para crear la imagen de Docker.
    - Ejecutar el comando `docker run -p 4000:80 NOMBRE` para ejecutar el contenedor de Docker. Para parar el contenedor de Docker, podemos hacer `docker stop <container_id>` y para volver a iniciarlo `docker start <container_id>`. (Para obtener el container_id, podemos hacer `docker ps` y copiar el id del contenedor de practica2, o ``NOMBRE``).
    - Para usar Redis podemos hacerlo de la misma forma que en el caso anterior.
  - *3.- Ejecutar nuestra aplicación, Redis, Grafana y el visualizador de contenedores con Docker Compose.*
    - Ejecutar el comando `docker-compose pull` para descargar las imágenes necesarias.
    - Ejecutar el comando `docker-compose up` para ejecutar el stack de Docker. Ahora nuestra aplicación está comunicada con Redis. Para parar el stack de Docker, podemos hacer `docker-compose down`. Esto creará un contenedor de Redis, un contenedor de Grafana y un contenedor de nuestro modelo de Machine Learning. También se creará un contenedor con el visualizador de contenedores.
  - *4.- Ejecutar todo con un stack de Docker **(RECOMENDADO)***.
    - Ejecutar el comando `docker stack deploy -c practica2/docker-compose.yml NOMBRE` para ejecutar el stack de Docker. Para parar el stack de Docker, podemos hacer `docker stack rm NOMBRE`. Esto creará un contenedor de Redis, un contenedor de Grafana, un contenedor del visualizador, y 5 replicas de nuestro modelo de Machine Learning. Con esto se establece un balanceo de carga entre los 5 contenedores de nuestra aplicación, que vuelven a ponerse en marcha en caso de que alguno de ellos falle automáticamente. (Asegurarse de usar `docker swarm init` para iniciar nuestra máquina de forma que pueda desplegar stacks de docker).

**Una vez que la aplicación está en marcha, podemos acceder a ella a través de ``localhost:80`` o ``localhost:4000``** (dependiendo de la opción que hayamos elegido). En caso de que hayamos usado Docker Compose o Docker Stack (o iniciado sus contenedores independientemente), podemos acceder a Grafana a través de ``localhost:3000`` y al visualizador de contenedores a través de ``localhost:8080``, así como hacer consultas a Redis a través de ``localhost:6379``.

## Práctica 3

El propósito principal de esta práctica es una introducción a Zookeeper. Crearemos varios nodos que enviarán una media de las temperaturas generadas a la API REST de la práctica 2.
El contenedor desarrollado se encuentra en Docker Hub
[DockerHub](https://hub.docker.com/repository/docker/aortegasdev/practica3)

**Instrucciones para ejecutar el código:**

- Se necesita **tener Docker Desktop instalado y Docker Engine en funcinamiento**.
- **Clonar el repositorio** o descargar los archivos en una carpeta local.
- **Abrir una terminal y situarse en la raíz del repositorio**.
- **Desplegar el stack de la práctica 2**.

```text
docker swarm init
docker stack deploy -c practica2/docker-compose.yml NOMBRE
```

- La app se encontrará en **``localhost:4000``**.
- **Desplejar un contenedor de Zookeeper**.

```text
docker run --name some-zookeeper --restart always -d -p 2181:2181 zookeeper
```

- **Desplegar uno o varios contenedores de la práctica 3, o bien ejecutar el script localmente**.

```text
docker run -e APP_ID=id aortegasdev/practica3
python -u practica3/zookeeper.py
```

> [!TIP]
> Para ejecutar varios contenedores de la práctica 3, ejecutaremos el comando `docker run -e APP_ID=id aortegasdev/practica3` varias veces, cambiando el id por uno diferente cada vez.
