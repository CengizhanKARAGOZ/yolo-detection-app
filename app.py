"""
YOLO11 Human & Vehicle Detection Application.

A Streamlit application for detecting humans and vehicles
in images and videos using YOLO11 model.
"""

import streamlit as st

from config import Config
from services import ModelService
from components import (
    SidebarComponent,
    ImageProcessorComponent,
    VideoProcessorComponent
)


class Application:
    """Main application class."""

    def __init__(self):
        self._configure_page()
        self._sidebar = SidebarComponent()
        self._model_service = ModelService()

    def _configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout="wide"
        )

    def run(self) -> None:
        """Run the application."""
        self._render_header()
        settings = self._sidebar.render()

        if not settings.model_path:
            st.warning("⚠️ Please select or upload a model from the sidebar.")
            return

        self._process_with_model(settings)

    def _render_header(self) -> None:
        """Render application header."""
        st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
        st.markdown("---")

    def _process_with_model(self, settings) -> None:
        """Load model and process user input."""
        try:
            model = self._model_service.load_model(settings.model_path)
            st.sidebar.success("✅ Model loaded successfully!")

            self._render_content(model, settings)

        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")

    def _render_content(self, model, settings) -> None:
        """Render main content based on user selection."""
        upload_type = st.radio(
            "Select File Type",
            ["📷 Image", "🎥 Video"],
            horizontal=True
        )

        if upload_type == "📷 Image":
            processor = ImageProcessorComponent(
                model,
                settings.confidence,
                settings.selected_classes
            )
            processor.render(Config.SUPPORTED_IMAGE_FORMATS)
        else:
            processor = VideoProcessorComponent(
                model,
                settings.confidence,
                settings.selected_classes
            )
            processor.render(Config.SUPPORTED_VIDEO_FORMATS)

        self._render_footer()

    def _render_footer(self) -> None:
        """Render application footer."""
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray;'>
                <p>YOLO11 Human & Vehicle Detection | Built with Streamlit</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    """Application entry point."""
    app = Application()
    app.run()


if __name__ == "__main__":
    main()