from dataclasses import dataclass
from typing import List, Optional

import streamlit as st

from config import Config
from services import ModelService


@dataclass
class SidebarSettings:
    model_path: Optional[str]
    confidence: float
    selected_classes: List[int]


class SidebarComponent:
    def __init__(self):
        self._model_service = ModelService()

    def render(self) -> SidebarSettings:
        with st.sidebar:
            st.header("⚙️ Settings")

            self._render_device_info()

            st.markdown("---")

            model_path = self._render_model_section()

            st.markdown("---")

            confidence = self._render_confidence_slider()
            selected_classes = self._render_class_selection()

            return SidebarSettings(
                model_path=model_path,
                confidence=confidence,
                selected_classes=selected_classes
            )

    def _render_device_info(self) -> None:
        device_info = self._model_service.get_device_info()

        if device_info['device'] == 'cuda':
            st.success(f"🚀 GPU: {device_info['name']}")
            st.caption(f"VRAM: {device_info['memory']}")
        else:
            st.warning("💻 Running on CPU")
            st.caption("CUDA not available")

    def _render_model_section(self) -> Optional[str]:
        model_source = st.radio(
            "Model Source",
            ["Default Model", "Upload Custom Model"]
        )

        if model_source == "Default Model":
            default_path = Config.DEFAULT_MODEL
            if default_path.exists():
                return str(default_path)
            else:
                st.error(f"⚠️ Model not found: {default_path}")
                st.info("Please add your model to the 'models' folder")
                return None

        uploaded_model = st.file_uploader("Model File (.pt)", type=["pt"])

        if uploaded_model:
            return self._model_service.save_uploaded_model(uploaded_model)

        st.warning("⚠️ Please upload a model file")
        return None

    def _render_confidence_slider(self) -> float:
        return st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=Config.DEFAULT_CONFIDENCE,
            step=0.05
        )

    def _render_class_selection(self) -> List[int]:
        st.markdown("### 🏷️ Detection Classes")

        selected_classes = []

        for class_id, detection_class in Config.DETECTION_CLASSES.items():
            label = f"{detection_class.emoji} {detection_class.name}"
            if st.checkbox(label, value=True, key=f"class_{class_id}"):
                selected_classes.append(class_id)

        return selected_classes