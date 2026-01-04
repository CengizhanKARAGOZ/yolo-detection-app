"""Application configuration constants."""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path


@dataclass(frozen=True)
class DetectionClass:
    """Represents a detection class with its properties."""
    id: int
    name: str
    emoji: str


class Config:
    """Application configuration."""

    # Paths
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"

    PAGE_TITLE = "YOLO11 Human & Vehicle Detection"
    PAGE_ICON = "🎯"

    DEFAULT_MODEL = MODELS_DIR / "best.pt"
    DEFAULT_CONFIDENCE = 0.5

    SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]
    SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "webm"]

    # Custom trained model classes: 0 = human, 1 = car
    DETECTION_CLASSES: Dict[int, DetectionClass] = {
        0: DetectionClass(0, "Human", "👤"),
        1: DetectionClass(1, "Car", "🚗"),
    }

    @classmethod
    def get_class_name(cls, class_id: int) -> str:
        """Get class name by ID."""
        detection_class = cls.DETECTION_CLASSES.get(class_id)
        return detection_class.name if detection_class else f"Class {class_id}"

    @classmethod
    def get_all_class_ids(cls) -> List[int]:
        """Get all available class IDs."""
        return list(cls.DETECTION_CLASSES.keys())