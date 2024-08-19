"""
Optical Character Recognition functions for gathering text information from OpenCV images.
"""

import cv2
from pytesseract import pytesseract


def extract_text(image: cv2.Mat, config_options=None) -> str:
    """
    Returns unmodified output as string from Tesseract OCR processing
    :param image:
    :param config_options:
    :return:
    """
    config = config_options if config_options else ""
    # By default, OpenCV stores images in BGR format and since pytesseract assumes RGB format,
    # we need to convert from BGR to RGB format/mode:
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(img_rgb, config=config)


def extract_data(image: cv2.Mat, config="--psm 6") -> str:
    """
    Returns result containing box boundaries, confidences, and other information.
    :param config:
    :param image:
    :return:
    """
    return pytesseract.image_to_data(image, config=config)


def extract_boxes(image: cv2.Mat, config_options=None):
    """
    Returns result containing recognized characters and their box boundaries
    :param image:
    :param config_options:
    :return:
    """
    config = config_options if config_options else ""
    return pytesseract.image_to_boxes(image, config=config)


def extract_osd(image: cv2.Mat, config_options=None):
    """
    Returns result containing information about orientation and script detection.
    :param image:
    :param config_options:
    :return:
    """
    config = config_options if config_options else ""
    return pytesseract.image_to_osd(image, config=config)


def detect_objects(image: cv2.Mat, cascade_file, scaleFactor=1.1, minNeighbors=5):
    cascade = cv2.CascadeClassifier(cascade_file)
    # Convert to grayscale for object detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect objects
    objects = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return objects
