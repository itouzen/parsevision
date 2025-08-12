import streamlit as st
from PIL import Image
import pdf2image
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)
from process.easy_ocr import process as easy_ocr_process
from process.tesseract_ocr import process as tesseract_ocr_process
import json
import cv2

st.title("Parse Vision")
st.write("Visualise what OCR libraries see in your documents")

uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

if uploaded_file is not None:
    images = convert_from_bytes(uploaded_file.getvalue())
    with st.spinner("Processing... this may take several seconds"):
        processed_images, easyocr_data = easy_ocr_process(images)
        processed_images2, tesseract_data = tesseract_ocr_process(images)
        combined_export = {
            "file_name": uploaded_file.name,
            "pages": easyocr_data + tesseract_data,
        }
        tab1, tab2 = st.tabs(["EasyOCR", "Tesseract"])
        with tab1:
            for i, image in enumerate(processed_images):
                # Convert BGR (cv2) to RGB for correct Streamlit display
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Page {i}", use_container_width=True)
            st.download_button(
                label="Download EasyOCR JSON",
                data=json.dumps({"file_name": uploaded_file.name, "pages": easyocr_data}, indent=2),
                file_name=f"{uploaded_file.name}_easyocr.json",
                mime="application/json",
            )
        with tab2:
            for i, image in enumerate(processed_images2):
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Page {i}", use_container_width=True)
            st.download_button(
                label="Download Tesseract JSON",
                data=json.dumps({"file_name": uploaded_file.name, "pages": tesseract_data}, indent=2),
                file_name=f"{uploaded_file.name}_tesseract.json",
                mime="application/json",
            )
        st.download_button(
            label="Download Combined JSON (EasyOCR + Tesseract)",
            data=json.dumps(combined_export, indent=2),
            file_name=f"{uploaded_file.name}_combined.json",
            mime="application/json",
        )
