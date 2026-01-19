# Face Recognition Manager

A comprehensive, GPU-accelerated face recognition system with real-time detection, video processing, and an intuitive GUI. Built with **InsightFace**, **MediaPipe**, and **ArcFace** models.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Supported-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)


https://github.com/user-attachments/assets/1d09f519-b88b-4144-9a4a-5eb9cc9dbc20


---

## ✨ Features

### 🎯 Multiple Recognition Backends
- **InsightFace** (Buffalo_L / Buffalo_S) - State-of-the-art accuracy with GPU acceleration
- **MediaPipe + ArcFace** (ResNet50 / MobileFaceNet) - Lightweight and fast

### 🎥 Real-Time Processing
- Live camera face detection and recognition
- Configurable FPS with skip-frame optimization
- **Multi-face tracking** with OpenCV CSRT/KCF trackers and IoU-based fallback
- **Identity caching** - recognized faces skip embedding for better performance

### 📹 Video Processing
- Process video files with face annotation
- Audio preservation (FFmpeg/MoviePy)
- Interactive labeling of unknown faces during processing
- Export processed videos with face labels

### 🧠 Smart Learning
- **Adaptive Learning** - Automatically update face templates for better recognition
- **Rolling Update** - Use a teacher model to train the student model
- **Quality-Aware Enrollment** - Only enroll high-quality, frontal faces

### 🔧 Advanced Features
- **Rotation Detection** - Detect faces at 90°, 180°, 270° orientations
- **Face Alignment** - ArcFace-standard 112x112 aligned crops
- **Smart k-NN Recognition** - Adaptive thresholding with ambiguity detection
- **Data Augmentation** - Automatic augmentation during enrollment

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GUI (DearPyGUI)                       │
├─────────────────────────────────────────────────────────────┤
│   Recognition Tab  │  Video Player Tab  │  Database Tab     │
├─────────────────────────────────────────────────────────────┤
│                     Processing Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  InsightFace │  │  MediaPipe   │  │   Face Quality   │   │
│  │   Backend    │  │   Backend    │  │     Checker      │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Multi-Object Tracker (CSRT/KCF/IoU)          │   │
│  │         + Identity Cache (skip re-embedding)         │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      FAISS Database                          │
│            (IndexIDMap2 + IndexFlatIP, 512D)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.12+
- CUDA 12.x (optional, for GPU acceleration)
- FFmpeg (optional, for audio in video processing; falls back to MoviePy if not available)

### Setup

```bash
# Clone the repository
git clone https://github.com/hiensumi/face-recognition-manager.git
cd face-recognition-manager

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Models
[Download the models directory](https://drive.google.com/file/d/1Izw6mlbwzTP2KFIu0of-GgNv6Kro5xIf/view)

Extract the archive and place the contents into the `./models/` directory so that the structure matches the [project structure](#-project-structure) below.

Models details:

| Model | Size | Description |
|-------|------|-------------|
| `buffalo_l` | ~275MB | High accuracy InsightFace model |
| `buffalo_s` | ~125MB | Fast InsightFace model |
| `arcface_w600k_r50.onnx` | ~166MB | ArcFace ResNet50 |
| `arcface_w600k_mbf.onnx` | ~13MB | ArcFace MobileFaceNet |
| `blaze_face_short_range.tflite` | ~200KB | MediaPipe face detector |
| `face_landmarker.task` | ~4MB | MediaPipe face landmarks |

---

## 🚀 Quick Start

```bash
# Run the GUI application
python gui_app.py
```

### First Steps

1. **Select a Profile**: Choose between `MediaPipe+ArcFace` (faster) or `InsightFace` (more accurate)
2. **Start Camera**: Click "Start Camera" in the Recognition tab
3. **Enroll Faces**: Click "Enroll Face" to add detected faces to the database
4. **Adjust Thresholds**: Use the sliders to tune recognition sensitivity

---

## 📖 Usage Guide

### Recognition Tab
- Real-time face detection and recognition from webcam
- Green box = Recognized, Red box = Unknown, Gray box = Low quality

### Video Player Tab
- **Process & Save**: Process a video file and save with face annotations
- **Playback Detection**: Enable real-time detection during video playback
- Unknown faces will prompt for labeling during processing

### Database Tab
- View all enrolled faces with face counts
- Delete or prune (reduce to 50) face embeddings per person

### Global Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Similarity Threshold | Minimum similarity for recognition | 0.70 |
| Detection Confidence | Face detector confidence threshold | 0.70 |
| Skip Frames | Process every N frames (0 = all) | 10 |
| Adaptive Threshold | Threshold for adaptive learning | 0.85 |
| Check Rotation | Try 90°/180° rotations for tilted faces | On |

---

## 🔬 Technical Details

### Face Tracking Pipeline (MediaPipe path)
```
Frame → Detection → Tracker Update → Identity Lookup
                         │                  │
                         ▼                  ▼
              ┌──────────────────┐   ┌─────────────┐
              │ OpenCV Tracker   │   │  Identity   │
              │ (CSRT→KCF→IoU)   │   │   Cache     │
              └──────────────────┘   └─────────────┘
                         │                  │
                         ▼                  ▼
              Match detections ←───── Known: use cache
              to existing tracks      Unknown: run embedding
```

- **Track Persistence**: Tracks survive up to 15 missed frames
- **Identity Caching**: Recognized faces skip embedding (major FPS boost)
- **Periodic Re-check**: Unknown faces re-scanned every 30 frames

### Face Embedding
- **Dimension**: 512-D normalized vectors
- **Similarity Metric**: Inner Product (cosine similarity after normalization)
- **Storage**: FAISS IndexIDMap2 with IndexFlatIP

### Recognition Algorithm
```python
# Smart k-NN with adaptive thresholding
1. Query FAISS for top-k (k=5) nearest neighbors
2. Convert distances to similarities: sim = (dist + 1) / 2
3. Apply adaptive threshold based on:
   - Clear winner detection (high gap between top-1 and rest)
   - Cluster detection (multiple faces of same person)
   - Ambiguity detection (similar scores for different people)
4. Return name with confidence score
```

### Face Quality Assessment (for Enrollment)
- Pose estimation (reject side profiles > 40°)
- Eye openness detection
- Blur detection (Laplacian variance)
- Brightness and contrast validation
- Landmark coverage check

---

## 📁 Project Structure

```
face_recognition_new/
├── gui_app.py              # Main GUI application
├── backend_insightface.py  # InsightFace wrapper
├── backend_mediapipe.py    # MediaPipe Tasks API wrapper
├── face_recog.py           # FAISS database & recognition logic
├── data_augmentation.py    # Augmentation pipeline
├── requirements.txt        # Python dependencies
├── models/                 # Model files
│   ├── buffalo_l/
│   ├── buffalo_s/
│   ├── arcface_w600k_r50.onnx
│   ├── arcface_w600k_mbf.onnx
│   ├── blaze_face_short_range.tflite
│   └── face_landmarker.task
├── database/               # Face embeddings storage
│   ├── buffalo_l/
│   ├── buffalo_s/
│   ├── arcface_w600k_r50/
│   └── arcface_w600k_mbf/
```

---

## ⚙️ Configuration

### GPU Acceleration

The system automatically uses CUDA if available. To force CPU:

```python
# In backend_insightface.py
self.app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 for CPU
```

### Augmentation Settings

```python
# In gui_app.py
self.aug_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.ISONoise(p=0.2),
])
```

---

## 🤝 Contributing

Contributions are welcome!
This is a personal side project built during free time (very much vibe-coded), so the code may not be perfect. Improvements, refactors, and suggestions are appreciated.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - State-of-the-art face analysis
- [MediaPipe](https://github.com/google/mediapipe) - Cross-platform ML solutions
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [DearPyGUI](https://github.com/hoffstadt/DearPyGui) - Fast Python GUI toolkit
- [Albumentations](https://github.com/albumentations-team/albumentations) - Image augmentation

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

