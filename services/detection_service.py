from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import tempfile
import subprocess
import shutil
import os

import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st

from config import Config


@dataclass
class DetectionResult:
    annotated_frame: np.ndarray
    statistics: Dict[str, int]


@dataclass
class VideoDetectionResult:
    output_path: str
    statistics: Dict[str, int]


class BaseDetector(ABC):
    def __init__(self, model: YOLO, confidence: float, classes: Optional[List[int]] = None):
        self._model = model
        self._confidence = confidence
        self._classes = classes

    @abstractmethod
    def detect(self, source) -> DetectionResult:
        pass

    def _run_inference(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            classes=self._classes if self._classes else None,
            verbose=False
        )

        annotated = results[0].plot()
        statistics = self._extract_statistics(results[0].boxes)

        return annotated, statistics

    def _extract_statistics(self, detections) -> Dict[str, int]:
        stats: Dict[str, int] = {}

        if detections is None or len(detections) == 0:
            return stats

        for box in detections:
            class_id = int(box.cls[0])
            class_name = Config.get_class_name(class_id)
            stats[class_name] = stats.get(class_name, 0) + 1

        return stats


class ImageDetector(BaseDetector):

    def detect(self, image: np.ndarray) -> DetectionResult:
        annotated, statistics = self._run_inference(image)
        return DetectionResult(annotated_frame=annotated, statistics=statistics)


class VideoDetector(BaseDetector):
    def detect(self, video_path: str) -> VideoDetectionResult:
        cap = cv2.VideoCapture(video_path)
        video_props = self._get_video_properties(cap)

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            temp_output,
            fourcc,
            video_props['fps'],
            (video_props['width'], video_props['height'])
        )

        total_stats = self._process_frames(cap, out, video_props['total_frames'])

        cap.release()
        out.release()

        final_output = self._convert_to_h264(temp_output)

        if os.path.exists(temp_output):
            os.unlink(temp_output)

        return VideoDetectionResult(output_path=final_output, statistics=total_stats)

    def _get_video_properties(self, cap: cv2.VideoCapture) -> Dict:
        return {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

    def _convert_to_h264(self, input_path: str) -> str:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                output_path
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            shutil.copy(input_path, output_path)

        return output_path

    def _process_frames(
        self,
        cap: cv2.VideoCapture,
        out: cv2.VideoWriter,
        total_frames: int
    ) -> Dict[str, int]:
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_stats: Dict[str, int] = {}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, frame_stats = self._run_inference(frame)
            out.write(annotated)

            self._merge_statistics(total_stats, frame_stats)

            frame_count += 1
            self._update_progress(progress_bar, status_text, frame_count, total_frames)

        progress_bar.empty()
        status_text.empty()

        return total_stats

    def _merge_statistics(self, total: Dict[str, int], frame_stats: Dict[str, int]) -> None:
        for class_name, count in frame_stats.items():
            total[class_name] = total.get(class_name, 0) + count

    def _update_progress(
        self,
        progress_bar,
        status_text,
        current: int,
        total: int
    ) -> None:
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"Processing: {current}/{total} frames")


class DetectorFactory:
    @staticmethod
    def create_image_detector(
        model: YOLO,
        confidence: float,
        classes: Optional[List[int]] = None
    ) -> ImageDetector:
        return ImageDetector(model, confidence, classes)

    @staticmethod
    def create_video_detector(
        model: YOLO,
        confidence: float,
        classes: Optional[List[int]] = None
    ) -> VideoDetector:
        return VideoDetector(model, confidence, classes)