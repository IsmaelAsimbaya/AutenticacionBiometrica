# Etapa de construcción
FROM python:3.9.19-slim AS builder

# Instalar dependencias de sistema necesarias para la construcción
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
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
    && rm -rf /var/lib/apt/lists/* /root/.cache

# Directorio de trabajo
WORKDIR /work

# Actualizar pip
RUN pip install --upgrade pip setuptools

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las dependencias en la etapa de construcción
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente de la aplicación
COPY src/ /work/

# Etapa de producción
FROM python:3.9.19-slim AS production

# Instalar dependencias de sistema necesarias para la ejecución
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenblas-dev \
    libfreetype6-dev \
    libatlas-base-dev \
    liblapack-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# Directorio de trabajo
WORKDIR /work

# Copiar las dependencias instaladas desde la etapa de construcción
COPY --from=builder /usr/local /usr/local

# Copiar el código fuente de la aplicación desde la etapa de construcción
COPY --from=builder /work /work

# Exponer el puerto para la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["fastapi", "dev", "app.py", "--host", "0.0.0.0", "--port", "8000"]

# Construccion de imagen dev en docker
# docker build -t app-pyautenbio .

# Despliegue de la iamgen dockerlis
# docker run -it -p 8000:8000 -v ${PWD}:/work pyautenbio sh