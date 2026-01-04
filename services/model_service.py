"""Model loading and management service."""

import tempfile

import streamlit as st
import torch
from ultralytics import YOLO


class ModelService:
    """Handles YOLO model loading and caching."""

    @staticmethod
    def get_device() -> str:
        """
        Detect available device (CUDA or CPU).

        Returns:
            Device string ('cuda' or 'cpu').
        """
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    @staticmethod
    def get_device_info() -> dict:
        """
        Get detailed device information.

        Returns:
            Dictionary with device details.
        """
        if torch.cuda.is_available():
            return {
                'device': 'cuda',
                'name': torch.cuda.get_device_name(0),
                'memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            }
        return {
            'device': 'cpu',
            'name': 'CPU',
            'memory': 'N/A'
        }

    @staticmethod
    @st.cache_resource
    def load_model(model_path: str) -> YOLO:
        """
        Load and cache YOLO model.

        Args:
            model_path: Path to the model file.

        Returns:
            Loaded YOLO model instance.
        """
        device = ModelService.get_device()
        model = YOLO(model_path)
        model.to(device)
        return model

    @staticmethod
    def save_uploaded_model(uploaded_file) -> str:
        """
        Save uploaded model file to temporary location.

        Args:
            uploaded_file: Streamlit uploaded file object.

        Returns:
            Path to the saved temporary file.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(uploaded_file.read())
            return tmp.name