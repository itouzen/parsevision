import easyocr
import certifi
import ssl
import numpy as np
import cv2
import os


# Set up SSL context for secure connections
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())


def process(images):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(["en"])
    processed_images = []

    for i, image in enumerate(images):
        img = np.array(image)

        # Read text from image
        easyocr_parser = reader.readtext(img, paragraph=False, width_ths=0.1)

        for result in easyocr_parser:
            # Extract bounding box points for detected text
            top_left = tuple(map(int, result[0][0]))
            bottom_right = tuple(map(int, result[0][2]))

            # Draw rectangle around detected text
            cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 3)

        processed_images.append(img)

    return processed_images
