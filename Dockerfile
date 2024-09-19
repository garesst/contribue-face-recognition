# Utiliza una imagen base de Python 3.9
FROM python:3.9-slim
LABEL authors='oscar gares'

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos requeridos en el contenedor
COPY . /app

# Actualiza e instala dependencias necesarias del sistema
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias del proyecto
RUN pip install --upgrade pip
RUN pip install deepface fastapi uvicorn[standard] opencv-python tf-keras python-multipart pillow facenet-pytorch

# Descargar los modelos principales de DeepFace
RUN python -c "from deepface import DeepFace; DeepFace.build_model('VGG-Face')"
RUN python -c "from deepface import DeepFace; DeepFace.build_model('DeepID')"
RUN python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"

# Expone el puerto donde la aplicación correrá
EXPOSE 8000

# Comando para correr la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
