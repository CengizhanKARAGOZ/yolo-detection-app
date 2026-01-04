# 🎯 YOLO11 Human & Vehicle Detection

A real-time object detection application built with Streamlit and YOLO11 for detecting humans and vehicles in images and videos.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Image Detection**: Upload and detect objects in images instantly
- **Video Detection**: Process videos frame-by-frame with real-time progress tracking
- **GPU Acceleration**: Automatic CUDA detection for faster inference
- **Adjustable Confidence**: Fine-tune detection sensitivity with confidence threshold slider
- **Class Filtering**: Toggle detection for specific classes (Human/Car)
- **Browser-Compatible Video**: Automatic H.264 encoding for seamless video playback
- **Download Results**: Export processed videos with bounding box annotations
- **Statistics Dashboard**: View detection counts per class

## 🎬 Demo

### Image Detection
Upload an image and see detected humans and vehicles with bounding boxes displayed side-by-side with the original.

### Video Detection
Upload a video, click "Start Detection", and watch the processed video with real-time annotations directly in your browser.

## 🏗️ Architecture

This project follows **Clean Code** principles and **SOLID** design patterns for maintainability and extensibility.

### Design Patterns

| Pattern | Implementation | Purpose |
|---------|----------------|---------|
| **Single Responsibility** | Separate modules for config, services, components | Each class has one job |
| **Open/Closed** | `BaseDetector` abstract class | Easy to extend without modifying existing code |
| **Dependency Inversion** | `DetectorFactory` | High-level modules don't depend on low-level modules |
| **Factory Pattern** | `DetectorFactory.create_*` | Centralized object creation |

### Layer Separation

```
┌─────────────────────────────────────────┐
│              Presentation               │
│     (app.py, components/*.py)           │
├─────────────────────────────────────────┤
│            Business Logic               │
│         (services/*.py)                 │
├─────────────────────────────────────────┤
│            Configuration                │
│            (config.py)                  │
└─────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git
- NVIDIA GPU with CUDA (optional, for faster inference)

### Step 1: Clone the Repository

```bash
git clone https://github.com/CengizhanKARAGOZ/yolo-detection-app.git
cd yolo-detection-app
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate yolo-detection
```

### Step 3: Install FFmpeg (for video processing)

```bash
conda install ffmpeg -c conda-forge
```

### Step 4: Add Your Model

Place your trained YOLO11 model file in the `models` folder:

```
models/
└── best.pt    <-- your model here
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 📖 Usage

### Image Detection

1. Select **"📷 Image"** from the file type options
2. Upload an image (supported formats: JPG, JPEG, PNG, BMP, WEBP)
3. View the original and detected image side-by-side
4. Check detection statistics below

### Video Detection

1. Select **"🎥 Video"** from the file type options
2. Upload a video (supported formats: MP4, AVI, MOV, MKV, WEBM)
3. Click **"▶️ Start Detection"** button
4. Wait for processing (progress bar shows current status)
5. Watch the processed video in browser
6. Download the result using **"⬇️ Download Processed Video"** button

### Sidebar Settings

- **Device Info**: Shows whether GPU (CUDA) or CPU is being used
- **Model Source**: Choose default model or upload custom model
- **Confidence Threshold**: Adjust detection sensitivity (0.0 - 1.0)
- **Detection Classes**: Toggle Human and/or Car detection

## ⚙️ Configuration

### Application Settings (`config.py`)
```python
# Example configuration in config.py
class Config:
    DEFAULT_MODEL = "models/best.pt"  # Default model path
    DEFAULT_CONFIDENCE = 0.5          # Default confidence threshold
    
    SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]
    SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "webm"]
```

### Streamlit Settings (`.streamlit/config.toml`)
```toml
[server]
maxUploadSize = 1024  # Max upload size in MB

[browser]
gatherUsageStats = false
```

### Adding New Detection Classes

To add new classes, edit the `DETECTION_CLASSES` dictionary in `config.py`:

```
yolo-detection-app/
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── models/
│   ├── .gitkeep
│   └── best.pt              # YOLO model file
├── components/              # UI Components
│   ├── __init__.py
│   ├── sidebar.py           # Sidebar settings
│   ├── image_processor.py   # Image detection UI
│   └── video_processor.py   # Video detection UI
├── services/                # Business Logic
│   ├── __init__.py
│   ├── model_service.py     # Model loading & device detection
│   └── detection_service.py # Detection algorithms
├── app.py                   # Application entry point
├── config.py                # Configuration constants
├── environment.yml          # Conda environment
├── requirements.txt         # Pip requirements (alternative)
├── .gitignore
├── LICENSE
└── README.md
```

## 🔧 Technical Details

### GPU vs CPU

The application automatically detects CUDA availability:

- **CUDA Available**: Model runs on GPU for faster inference
- **CUDA Not Available**: Falls back to CPU

Device info is displayed in the sidebar.

### Video Processing Pipeline

```
Input Video → Frame Extraction → YOLO Inference → Annotation → 
XVID Encoding → FFmpeg H.264 Conversion → Browser Playback
```

### Supported Classes

| Class | ID | Description |
|-------|-----|-------------|
| Human | 0 | Pedestrians, people |
| Car | 1 | All vehicles (cars, trucks, buses, motorcycles) |

> **Note**: This model was custom-trained with 2 classes. The "Car" class includes all vehicle types.

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/CengizhanKARAGOZ/yolo-detection-app.git

# Create environment
conda env create -f environment.yml
conda activate yolo-detection

# Run in development
streamlit run app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO implementation
- [Streamlit](https://streamlit.io/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [FFmpeg](https://ffmpeg.org/) - Video processing

---

<p align="center">
  Made with ❤️ using YOLO11 and Streamlit
</p>

<p align="center">
  ⭐ Star this repository if you find it helpful!
</p>