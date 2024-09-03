# Construcción
FROM python:3.9.19-slim AS dev

# Instalar dependencias de sistema necesarias para la construcción
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libboost-all-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenblas-dev \
    libfreetype6-dev \
    libatlas-base-dev \
    liblapack-dev \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /work

# Actualizar pip
RUN /usr/local/bin/python -m pip install --upgrade pip setuptools

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las dependencias en la etapa de construcción
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente de la aplicación
COPY src/ /work/

# Exponer el puerto para la aplicación
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["fastapi", "dev", "app.py", "--host", "0.0.0.0", "--port", "8080"]

# Construccion de imagen dev en docker
# docker build --target dev . -t pyautenbio

# Despliegue de la iamgen dockerlis
# docker run -it -p 8000:8000 -v ${PWD}:/work pyautenbio sh