import easyocr
import cv2
import numpy as np
from paddleocr import PaddleOCR

from processing.image import preprocess_image


def verifyOcr(imageOne: np.array):
    # Guardar la imagen original
    cv2.imwrite("ocr.jpg", imageOne)

    # Crear una copia para dibujar rectángulos
    image_with_boxes = imageOne.copy()

    # Iniciar el lector de OCR
    reader = easyocr.Reader(['es', 'en'], gpu=True)

    # Realizar la lectura de texto en la imagen
    result = reader.readtext(imageOne)

    resulText = ''

    # Dibujar rectángulos alrededor de cada cuadro delimitador y acumular el texto
    for bbox, text, score in result:
        # bbox contiene 4 puntos (x, y) que forman un cuadrado alrededor del texto
        top_left = tuple([int(val) for val in bbox[0]])  # Esquina superior izquierda
        bottom_right = tuple([int(val) for val in bbox[2]])  # Esquina inferior derecha

        # Dibujar el rectángulo en la imagen
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)

        # Añadir el texto detectado
        resulText += text + ' '

    # Guardar la imagen con los rectángulos
    cv2.imwrite("ocr2.jpg", image_with_boxes)

    # Devolver el texto encontrado
    return resulText


def verifyOcrWithPaddle(imageOne: np.array):
    dir = preprocess_image(imageOne)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    img = cv2.imread(dir[0])
    result = ocr.ocr(dir[0])

    resulText = ''

    for line in result:
        for text_info in line:
            if len(text_info) == 2:
                bbox, text_info = text_info
                text = text_info[0]
            elif len(text_info) == 3:
                bbox, text, score = text_info
            else:
                continue

            resulText += text + ' '
            bbox = np.array(bbox).astype(np.int32)
            cv2.polylines(img, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(img, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(dir[1], img)

    return resulText
