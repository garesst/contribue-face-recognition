import easyocr
import cv2
import re
import numpy as np
from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode

from processing.image import preprocess_image


def verifyOcr(imageOne: np.array):
    # Guardar la imagen original
    dir = preprocess_image(imageOne)
    img = cv2.imread(dir[0])

    # Crear una copia para dibujar rectángulos
    image_with_boxes = img

    # Iniciar el lector de OCR
    reader = easyocr.Reader(['en', 'es'], gpu=True)

    # Realizar la lectura de texto en la imagen
    result = reader.readtext(dir[0])

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
    cv2.imwrite(dir[1], image_with_boxes)

    # Devolver el texto encontrado
    return resulText


def verifyOcrWithPaddle(imageOne: np.array, type: str):
    dir = preprocess_image(imageOne)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    img = cv2.imread(dir[0])
    result = ocr.ocr(dir[0])

    resulText = ''
    ocr_result = []  # Aquí se almacenarán las cajas, texto y confianza

    for line in result:
        for text_info in line:
            if len(text_info) == 2:
                bbox, text_info = text_info
                text = text_info[0]
                confidence = text_info[1] if len(text_info) > 1 else None
            elif len(text_info) == 3:
                bbox, text, confidence = text_info
            else:
                continue

            resulText += text + ' '
            bbox = np.array(bbox).astype(np.int32)
            ocr_result.append({"bbox": bbox.tolist(), "text": text, "confidence": confidence})

            # Dibujar las cajas y texto en la imagen
            cv2.polylines(img, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(img, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Guardar la imagen procesada
    cv2.imwrite(dir[1], img)

    # Extraer los campos utilizando resulText
    # datos_extraidos = extract_fields(resulText)
    # result_lines = [f"{campo}: {valor}" for campo, valor in datos_extraidos.items()]
    # result2 = "\n".join(result_lines)
    datos_extraidos2 = associate_fields_front(ocr_result) if type == 'front' else associate_fields_back(ocr_result)

    # Devuelve tanto el resultado procesado como el OCR completo
    return datos_extraidos2


def read_pdf417(imageOne: np.array):
    dir = preprocess_image(imageOne)
    img = cv2.imread(dir[0])
    codigos = decode(img)

    for codigo in codigos:
        if codigo.type == 'PDF417':  # Verifica que el tipo sea PDF417
            datos = codigo.data.decode('utf-8')  # Decodifica los datos
            print(f"Datos del PDF417: {datos}")
            return datos

    print("No se detectó un código PDF417 en la imagen.")
    return None


def extract_ID(image_path: str):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Usar PaddleOCR con detección de ángulos
    image = cv2.imread(image_path)

    # Realizar OCR en la imagen
    result = ocr.ocr(image, cls=True)

    # Verificar si el resultado tiene datos y procesarlo
    if result and len(result[0]) > 0:
        text_blocks = []
        for line in result[0]:  # El resultado está en el primer elemento de la lista
            bbox = line[0]
            text = line[1][0] if len(line[1]) > 0 else ""
            score = line[1][1] if len(line[1]) > 1 else 0
            text_blocks.append((bbox, text, score))

        return text_blocks
    else:
        print("No se detectó texto en la imagen.")
        return []


def find_closest_value(field_index, text_blocks):
    field_bbox, field_text, _ = text_blocks[field_index]

    field_x1, field_y1 = field_bbox[0]  # Coordenada superior izquierda del campo
    field_x2, field_y2 = field_bbox[2]  # Coordenada inferior derecha del campo

    closest_value = None
    min_distance = float('inf')

    for i, (bbox, text, score) in enumerate(text_blocks):
        if i == field_index:
            continue  # Saltar el campo actual

        value_x1, value_y1 = bbox[0]
        value_x2, value_y2 = bbox[2]

        # Determinar si el bloque de texto está a la derecha o debajo del campo
        if (value_x1 > field_x2) or (value_y1 > field_y2):
            # Calcular la distancia entre el campo y el valor
            distance = (value_x1 - field_x2) ** 2 + (value_y1 - field_y2) ** 2
            if distance < min_distance:
                min_distance = distance
                closest_value = text

    return closest_value


def extract_fields(ocr_text: str):
    # Diccionario para guardar los campos
    extracted_data = {
        "Nombres": "",
        "Apellidos": "",
        "Género": "",
        "Lugar y Fecha de Nacimiento": "",
        "Lugar y Fecha de Expedición": "",
        "Fecha de Expiración": "",
        "Número de Identificación": "",
    }

    # Usar expresiones regulares para extraer los datos
    apellidos_match = re.search(r"Apellidos/Surname\s+([A-Za-z\s]+)", ocr_text)
    if apellidos_match:
        extracted_data["Apellidos"] = apellidos_match.group(1).strip()

    nombres_match = re.search(r"Nombre /Given Names\s+([A-Za-z\s]+)", ocr_text)
    if nombres_match:
        extracted_data["Nombres"] = nombres_match.group(1).strip()

    genero_match = re.search(r"Genero/Gender\s+([A-Za-z\s]+)", ocr_text)
    if genero_match:
        extracted_data["Género"] = genero_match.group(1).strip()

    nacimiento_match = re.search(r"Lugar y Fecha de Nacimiento /Place and Date of Birth\s+([A-Za-z\s,]+)\s+([\d/]+)",
                                 ocr_text)
    if nacimiento_match:
        extracted_data[
            "Lugar y Fecha de Nacimiento"] = f"{nacimiento_match.group(1).strip()}, {nacimiento_match.group(2).strip()}"

    expedicion_match = re.search(r"Lugar y Fecha de Expedicion/Place and Date of Issuance\s+([A-Za-z\s,]+)\s+([\d/]+)",
                                 ocr_text)
    if expedicion_match:
        extracted_data[
            "Lugar y Fecha de Expedición"] = f"{expedicion_match.group(1).strip()}, {expedicion_match.group(2).strip()}"

    fecha_expiracion = re.search(r"Fecha de Expiracion /Date of Expiratio\s+([\d/]+)", ocr_text)
    if fecha_expiracion:
        extracted_data["Fecha de Expiración"] = fecha_expiracion.group(1).strip()

    unique_id = re.search(r"Unique lOnumber\s+([\d\-]+)", ocr_text)
    if unique_id:
        extracted_data["Número de Identificación"] = unique_id.group(1).strip()

    return extracted_data


def associate_fields_front(ocr_result):
    # Diccionario para los datos asociados
    extracted_fields = {}

    for i, item in enumerate(ocr_result):
        bbox = item["bbox"]
        text = item["text"]
        confidence = item["confidence"]

        # Aquí puedes procesar los textos para asociarlos a campos específicos
        if "Apellidos" in text:
            extracted_fields["Apellidos"] = ocr_result[i + 1]["text"] if i + 1 < len(ocr_result) else ""
        elif "Nombre" in text and "Names" in text:
            extracted_fields["Nombres"] = ocr_result[i + 1]["text"] if i + 1 < len(ocr_result) else ""
        elif "Genero" in text:
            extracted_fields["Género"] = ocr_result[i + 1]["text"] if i + 1 < len(ocr_result) else ""
        elif "Lugar y Fecha de Nacimiento" in text:
            extracted_fields["Lugar y Fecha de Nacimiento"] = (
                ocr_result[i + 1]["text"] if i + 1 < len(ocr_result) else ""
            )
        elif "Lugar y Fecha de Expedicion" in text:
            extracted_fields["Lugar y Fecha de Expedición"] = (
                                                                  ocr_result[i + 1]["text"] if i + 1 < len(
                                                                      ocr_result) else ""
                                                              ) + ' ' + (
                                                                  ocr_result[i + 2]["text"] if i + 2 < len(
                                                                      ocr_result) else ""
                                                              )
        elif "Fecha de Expiracion" in text:
            extracted_fields["Fecha de Expiración"] = ocr_result[i + 1]["text"] if i + 1 < len(ocr_result) else ""
        elif "Unique lOnumber" in text or "Número" in text:
            extracted_fields["Número de Identificación"] = ocr_result[i + 1]["text"] if i + 1 < len(ocr_result) else ""

    return extracted_fields


def associate_fields_back(ocr_result):
    # Diccionario para los datos asociados
    extracted_fields = {}

    for i, item in enumerate(ocr_result):
        bbox = item["bbox"]
        text = item["text"]
        confidence = item["confidence"]

        # Aquí puedes procesar los textos para asociarlos a campos específicos
        if "Departamento" in text and "State" in text:
            vartem = ocr_result[i + 2]["text"] if i + 2 < len(ocr_result) else ""
            varnum = 3 if 'nit' in vartem else 2
            extracted_fields["Departamento"] = ocr_result[i + varnum]["text"] if i + varnum < len(ocr_result) else ""
        elif "Municipio" in text and "City" in text:
            extracted_fields["Municipio"] = ocr_result[i + 2]["text"] if i + 2 < len(ocr_result) else ""
        elif "Estado familiar" in text:
            extracted_fields["Estado familiar"] = ocr_result[i + 2]["text"] if i + 2 < len(ocr_result) else ""
        elif "Profesion" in text:
            extracted_fields["Profesion"] = ocr_result[i + 1]["text"] if i + 1 < len(ocr_result) else ""

    return extracted_fields