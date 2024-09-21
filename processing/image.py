import uuid

import cv2
import numpy as np


def preprocess_image(imageOne: np.array):
    uuid_name = f"{uuid.uuid4()}"
    type_image = ".jpg"
    temp_filename = 'image/' + uuid_name
    cv2.imwrite(temp_filename + "_origin" + type_image, imageOne)
    gray = cv2.cvtColor(imageOne, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # gray = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    scale_percent = 200
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    image_with_boxes2 = gray.copy()
    cv2.imwrite(temp_filename + type_image, image_with_boxes2)
    return temp_filename + type_image, temp_filename + "_box" + type_image
