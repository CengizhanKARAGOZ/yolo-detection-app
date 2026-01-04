from typing import Dict

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from services import DetectorFactory, DetectionResult
from ultralytics import YOLO


class ImageProcessorComponent:
    def __init__(self, model: YOLO, confidence: float, classes: list):
        self._detector = DetectorFactory.create_image_detector(model, confidence, classes)

    def render(self, supported_formats: list) -> None:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=supported_formats
        )

        if not uploaded_file:
            return

        self._process_and_display(uploaded_file)

    def _process_and_display(self, uploaded_file) -> None:
        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file)

        with col1:
            st.subheader("📥 Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🎯 Detection Result")
            result = self._detect(image)
            st.image(result.annotated_frame, use_container_width=True)

        self._display_statistics(result.statistics)

    def _detect(self, image: Image.Image) -> DetectionResult:
        with st.spinner("Detecting..."):
            img_array = self._prepare_image(image)
            result = self._detector.detect(img_array)
            result.annotated_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
            return result

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        img_array = np.array(image)

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        return img_array

    def _display_statistics(self, statistics: Dict[str, int]) -> None:
        if not statistics:
            st.info("ℹ️ No detections found for selected classes.")
            return

        st.markdown("---")
        st.subheader("📊 Detection Statistics")

        cols = st.columns(len(statistics))
        for i, (class_name, count) in enumerate(statistics.items()):
            with cols[i]:
                st.metric(class_name, count)