import easyocr
import certifi
import ssl
import numpy as np
import cv2


# Fix: previously we overwrote ssl._create_default_https_context with an SSLContext
# instance, causing TypeError: 'SSLContext' object is not callable when libraries
# attempted to call it. Instead provide a factory function returning the context.
def _patched_create_default_https_context(*args, **kwargs):  # noqa: D401
    """Return a default SSL context using certifi bundle (patch for EasyOCR downloads)."""
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = _patched_create_default_https_context  # type: ignore


def process(images):
    """Run EasyOCR on a list of PIL images.

    Returns a tuple (processed_images, ocr_pages) where:
      processed_images: list[np.ndarray] with drawn rectangles
      ocr_pages: list[dict] structured OCR output per page
    """
    reader = easyocr.Reader(["en"])  # Initialize EasyOCR reader once
    processed_images = []
    ocr_pages = []

    for page_index, image in enumerate(images):
        img = np.array(image)
        page_items = []

        # Perform OCR
        easyocr_parser = reader.readtext(img, paragraph=False, width_ths=0.1)

        for result in easyocr_parser:
            bbox, text, confidence = result
            # bbox is a list of 4 points (each point is [x,y])
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 3)
            page_items.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": {
                        "top_left": list(map(int, bbox[0])),
                        "top_right": list(map(int, bbox[1])),
                        "bottom_right": list(map(int, bbox[2])),
                        "bottom_left": list(map(int, bbox[3])),
                    },
                }
            )

        processed_images.append(img)
        ocr_pages.append(
            {
                "engine": "easyocr",
                "page": page_index,
                "items": page_items,
            }
        )

    return processed_images, ocr_pages
