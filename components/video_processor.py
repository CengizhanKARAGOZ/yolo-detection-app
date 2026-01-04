from typing import Dict
import tempfile
import os

import streamlit as st
from ultralytics import YOLO

from services import DetectorFactory, VideoDetectionResult


class VideoProcessorComponent:
    def __init__(self, model: YOLO, confidence: float, classes: list):
        self._detector = DetectorFactory.create_video_detector(model, confidence, classes)
        self._temp_files = []

    def render(self, supported_formats: list) -> None:
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=supported_formats
        )

        if not uploaded_file:
            return

        input_path = self._save_uploaded_video(uploaded_file)
        self._process_and_display(input_path)
        self._cleanup_temp_files()

    def _save_uploaded_video(self, uploaded_file) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            self._temp_files.append(tmp.name)
            return tmp.name

    def _process_and_display(self, input_path: str) -> None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Original Video")
            st.video(input_path)

        with col2:
            st.subheader("🎯 Detection Result")
            self._render_detection_ui(input_path)

    def _render_detection_ui(self, input_path: str) -> None:
        if not st.button("▶️ Start Detection", type="primary", use_container_width=True):
            st.info("👆 Click the button to start detection")
            return

        result = self._detector.detect(input_path)
        self._temp_files.append(result.output_path)

        self._display_result_video(result)
        self._display_statistics(result.statistics)

    def _display_result_video(self, result: VideoDetectionResult) -> None:
        st.video(result.output_path)

        with open(result.output_path, "rb") as f:
            st.download_button(
                "⬇️ Download Processed Video",
                f,
                file_name="detected_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )

    def _display_statistics(self, statistics: Dict[str, int]) -> None:
        if not statistics:
            return

        st.markdown("---")
        st.subheader("📊 Total Detection Statistics")
        st.caption("(Total detections across all frames)")

        cols = st.columns(min(len(statistics), 5))
        for i, (class_name, count) in enumerate(statistics.items()):
            with cols[i % 5]:
                st.metric(class_name, count)

    def _cleanup_temp_files(self) -> None:
        for file_path in self._temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception:
                pass
        self._temp_files.clear()