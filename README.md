# 🤖 Contribue-Face-Recognition
### Servicios de Verificación Facial y OCR para DUI v0.1 Pasaporte v0.2

API RESTful desarrollada en **Python** con **FastAPI**, diseñada para integrarse con una aplicación de verificación KYC. Ofrece servicios de:
- Reconocimiento facial
- Detección de spoofing (comprobación de vida)
- Extracción de datos mediante OCR
- Comparación facial entre imágenes

Ideal para validar documentos de identidad salvadoreño y verificar la autenticidad del titular.

---

## 📦 Tecnologías Usadas

- **FastAPI** – Framework para construir APIs rápidas con validación automática
- **DeepFace** – Análisis facial y comparación de rostros
- **FaceNet-PyTorch** – Para detección y embeddings faciales
- **OpenCV** – Preprocesamiento de imágenes
- **EasyOCR + PaddleOCR** – Reconocimiento óptico de texto avanzado
- **Pillow** – Manipulación de imágenes
- **Pyzbar** – Lectura de códigos QR y de barras
- **TF-Keras / TensorFlow** – Modelos de aprendizaje profundo
- **Uvicorn** – Servidor ASGI para correr la API
- **Python-Multipart** – Para soportar carga de archivos
- **PaddlePaddle** – Motor detrás de PaddleOCR

---

## 🔌 Endpoints Disponibles

### 1. `/recognize-face/` – Análisis facial y detección de spoofing

#### POST

**Descripción:** Analiza una imagen de rostro para detectar si es una imagen real o un intento de fraude (spoof).

**URL:** `POST http://127.0.0.1:8000/recognize-face/`

**Body (JSON):**
```json
{
  "image_base64": "string (base64)"
}
```
**Respuesta (Éxito):**
```json
{
  "deepface_analysis": {
    "Edad": Number,
    "Genero": "string",
    "Emocion": "string",
    "Raza": "string"
  }
}
```
**Respuesta (Error de spoof):**
```json
{
  "deepface_analysis": {
    "error": "Anti-spoofing check failed. Please make sure you are not using a spoofed image.",
    "details": "Spoof detected in the given image."
  }
}
```

### 2. `/ocr2/`  – Extracción de información del documento

#### POST

**Descripción:** Extrae información del documento de identidad (frontal y trasero).

**URL:** `POST http://127.0.0.1:8000/ocr2/`

**Body (JSON):**
```json
{
  "front_image": "base64_string",
  "back_image": "base64_string"
}
```
**Respuesta (Ejemplo):**
```json
{
  "deepface_analysis": {
    "Edad": 31,
    "Genero": "Man",
    "Emocion": "fear",
    "Raza": "white",
    "face_confidence": 0.94
  },
  "front": {
    "Apellidos": "PEREZ ALVARADO",
    "Nombres": "JUAN PABLO",
    "Género": "M",
    "Lugar y Fecha de Nacimiento": "SAN SALVADOR,SAN SALVADOR 31/05/1983",
    "Lugar y Fecha de Expedición": "SAN SALVADOR,SAN SALVADOR 29/04/2023",
    "Fecha de Expiración": "28/04/2031",
    "Número de Identificación": "00000000-0"
  },
  "back": {
    "Departamento": "SAN SALVADOR",
    "Municipio": "SAN SALVADOR",
    "Estado familiar": "SOLTERO(A)",
    "Profesion": "ESTUDIANTE"
  },
  "pdf14": null
}
```

### 3. `/face/`  – Comparación facial entre dos imágenes

#### POST

**Descripción:** Compara dos imágenes para verificar si pertenecen a la misma persona.

**URL:** `POST http://127.0.0.1:8000/face/`

**Body (JSON):**
```json
{
  "image_one": "base64_string",
  "image_two": "base64_string"
}
```
**Respuesta (Ejemplo):**
```json
{
  "verified": false,
  "distance": 0.9103773718256487,
  "threshold": 0.68,
  "model": "VGG-Face",
  "detector_backend": "opencv",
  "similarity_metric": "cosine",
  "facial_areas": {
    "img1": { ... },
    "img2": { ... }
  },
  "time": 2.41
}
```

---

## ⚙️ Instalación y Ejecución

1. Clonar repositorio
2. Instalar dependencias `pip install -r requirements.txt`
3. Iniciar servidor `uvicorn main:app --reload`