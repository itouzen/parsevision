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
import pandas as pd

st.title("Parse Vision")
st.write("Visualise what OCR libraries see in your documents and export per-page OCR JSON.")

uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

if uploaded_file is not None:
    images = convert_from_bytes(uploaded_file.getvalue())
    with st.spinner("Processing... this may take several seconds"):
        processed_images, easyocr_data = easy_ocr_process(images)
        processed_images2, tesseract_data = tesseract_ocr_process(images)

    tab1, tab2 = st.tabs(["EasyOCR", "Tesseract"])
    with tab1:
        for i, image in enumerate(processed_images):
            page_dict = next((p for p in easyocr_data if p.get("page") == i), None)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Page {i}", use_container_width=True)
            if page_dict:
                st.download_button(
                    label=f"Download Page {i} JSON",
                    data=json.dumps(page_dict, indent=2),
                    file_name=f"{uploaded_file.name}_easyocr_page_{i}.json",
                    mime="application/json",
                    key=f"easyocr_page_dl_{i}",
                )
            st.divider()
        if st.button("Show EasyOCR DataFrame", key="easyocr_df_btn"):
            # Flatten EasyOCR data into rows
            rows = []
            for page in easyocr_data:
                page_num = page.get("page")
                for idx, item in enumerate(page.get("items", [])):
                    bbox = item.get("bbox", {})
                    rows.append(
                        {
                            "engine": "easyocr",
                            "page": page_num,
                            "item_index": idx,
                            "text": item.get("text"),
                            "confidence": item.get("confidence"),
                            "top_left_x": (bbox.get("top_left") or [None, None])[0],
                            "top_left_y": (bbox.get("top_left") or [None, None])[1],
                            "top_right_x": (bbox.get("top_right") or [None, None])[0],
                            "top_right_y": (bbox.get("top_right") or [None, None])[1],
                            "bottom_right_x": (bbox.get("bottom_right") or [None, None])[0],
                            "bottom_right_y": (bbox.get("bottom_right") or [None, None])[1],
                            "bottom_left_x": (bbox.get("bottom_left") or [None, None])[0],
                            "bottom_left_y": (bbox.get("bottom_left") or [None, None])[1],
                        }
                    )
            if rows:
                df_easy = pd.DataFrame(rows)
                st.dataframe(df_easy, use_container_width=True)
                st.download_button(
                    label="Download EasyOCR CSV",
                    data=df_easy.to_csv(index=False),
                    file_name=f"{uploaded_file.name}_easyocr.csv",
                    mime="text/csv",
                    key="easyocr_csv_dl",
                )
            else:
                st.info("No EasyOCR text items extracted.")
    with tab2:
        for i, image in enumerate(processed_images2):
            page_dict = next((p for p in tesseract_data if p.get("page") == i), None)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Page {i}", use_container_width=True)
            if page_dict:
                st.download_button(
                    label=f"Download Page {i} JSON",
                    data=json.dumps(page_dict, indent=2),
                    file_name=f"{uploaded_file.name}_tesseract_page_{i}.json",
                    mime="application/json",
                    key=f"tesseract_page_dl_{i}",
                )
            st.divider()
        if st.button("Show Tesseract DataFrame", key="tesseract_df_btn"):
            rows = []
            for page in tesseract_data:
                page_num = page.get("page")
                for idx, item in enumerate(page.get("items", [])):
                    rows.append(
                        {
                            "engine": "tesseract",
                            "page": page_num,
                            "item_index": idx,
                            "text": item.get("text"),
                            "confidence": item.get("confidence"),
                        }
                    )
            if rows:
                df_tess = pd.DataFrame(rows)
                st.dataframe(df_tess, use_container_width=True)
                st.download_button(
                    label="Download Tesseract CSV",
                    data=df_tess.to_csv(index=False),
                    file_name=f"{uploaded_file.name}_tesseract.csv",
                    mime="text/csv",
                    key="tesseract_csv_dl",
                )
            else:
                st.info("No Tesseract text items extracted.")
