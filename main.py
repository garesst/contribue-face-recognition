from deepface import DeepFace
from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()


# Modelo Pydantic para recibir base64 en el cuerpo de la solicitud
class ImageBase64(BaseModel):
    image_base64: str


class ImageCompare(BaseModel):
    image_one: str
    image_two: str


def decode_base64_image(base64_string: str) -> np.array:
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# Función para reconocimiento facial usando DeepFace
def recognize_face_with_deepface(image: np.array):
    try:
        # Guardar temporalmente la imagen para ser usada por DeepFace
        cv2.imwrite("temp_image.jpg", image)

        # Usar DeepFace para análisis facial
        result = DeepFace.analyze(img_path="temp_image.jpg", actions=['age', 'gender', 'emotion', 'race'],
                                  enforce_detection=False, anti_spoofing=True)

        return result

    except Exception as e:
        # Devolver un mensaje de error en inglés si ocurre un problema
        return {
            "error": "Anti-spoofing check failed. Please make sure you are not using a spoofed image.",
            "details": str(e)  # Detalles opcionales del error
        }


def verify(imageOne: np.array, imageTwo: np.array):
    # Guardar temporalmente la imagen para ser usada por DeepFace
    cv2.imwrite("temp_image1.jpg", imageOne)
    cv2.imwrite("temp_image1.jpg", imageTwo)

    # Usar DeepFace para análisis facial
    result = DeepFace.verify(img1_path=imageOne, img2_path=imageTwo)

    return result


# Endpoint para procesar una imagen en base64
@app.post("/recognize-face/")
async def recognize_image(image_data: ImageBase64):
    # Convertir la imagen de base64 a numpy array para OpenCV
    image = decode_base64_image(image_data.image_base64)

    # Reconocimiento facial con DeepFace
    deepface_result = recognize_face_with_deepface(image)

    # Responder con los resultados de DeepFace y anti-spoofing
    return {
        "deepface_analysis": deepface_result
    }


# Endpoint para procesar una imagen en base64
@app.post("/face/")
async def face(data: ImageCompare):
    # Convertir la imagen de base64 a numpy array para OpenCV
    imageOne = decode_base64_image(data.image_one)
    imageTwo = decode_base64_image(data.image_two)

    # Reconocimiento facial con DeepFace
    deepface_result = verify(imageOne, imageTwo)

    # Responder con los resultados de DeepFace y anti-spoofing
    return deepface_result
