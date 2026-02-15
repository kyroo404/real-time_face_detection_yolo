# 🚀 Real-Time Face Detection with YOLOv26

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv26](https://img.shields.io/badge/YOLOv26-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

A high-performance real-time face detection application powered by **Ultralytics YOLOv26**, the latest NMS-free YOLO architecture. This project provides out-of-the-box functionality for detecting faces from webcam feeds or video files with configurable parameters and training scripts for custom face datasets.

---

## ✨ Features

- 🎥 **Real-time face detection** from webcam or video files
- ⚡ **YOLOv26 NMS-free architecture** for ultra-fast inference
- 🎯 **Configurable confidence threshold** for detection sensitivity
- 📊 **Live FPS counter** for performance monitoring
- 💾 **Video output saving** with configurable quality
- 🔧 **Multiple YOLO model sizes** (nano, small, medium, large, x-large)
- 📦 **Training script included** for fine-tuning on custom face datasets
- 🎨 **Clean, annotated bounding boxes** with confidence scores
- 🐍 **Pure Python implementation** with minimal dependencies

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Ultralytics YOLOv26** (≥8.4.0) - NMS-free object detection
- **OpenCV** (≥4.8.0) - Video processing and display
- **NumPy** (≥1.24.0) - Numerical operations

---

## 📋 Prerequisites

- Python 3.10 or higher
- Webcam (for real-time detection) or video files
- CUDA-capable GPU (optional, but recommended for better performance)

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/kyroo404/real-time_face_detection_yolo.git
cd real-time_face_detection_yolo
```

### 2. Create a virtual environment (recommended)

```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first time you run the detection script, YOLOv26 will automatically download the pretrained model weights (~6MB for nano model).

---

## 💻 Usage

### Basic Usage (Webcam)

Run face detection with your default webcam:

```bash
python detect_faces.py
```

Press **`q`** to quit the application.

### Video File Detection

Detect faces in a video file:

```bash
python detect_faces.py --source path/to/video.mp4
```

### Adjustable Confidence Threshold

Set a custom confidence threshold (0.0 to 1.0):

```bash
python detect_faces.py --conf 0.5
```

### Different Model Sizes

Choose from various YOLOv26 model sizes for speed/accuracy trade-offs:

```bash
# Nano (fastest, default)
python detect_faces.py --model yolo26n.pt

# Small
python detect_faces.py --model yolo26s.pt

# Medium
python detect_faces.py --model yolo26m.pt

# Large
python detect_faces.py --model yolo26l.pt

# Extra Large (most accurate)
python detect_faces.py --model yolo26x.pt
```

### Save Output Video

Save the detection results to a video file:

```bash
python detect_faces.py --source video.mp4 --save --output output/result.mp4
```

### Combined Example

```bash
python detect_faces.py --source webcam.mp4 --conf 0.6 --model yolo26s.pt --save --thickness 3
```

---

## ⚙️ Configuration / CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | `0` | Video source: `0` for webcam, or path to video file |
| `--conf` | float | `0.40` | Confidence threshold for detections (0.0-1.0) |
| `--model` | str | `yolo26n.pt` | YOLOv26 model variant (n/s/m/l/x) |
| `--save` | flag | `False` | Save output video to disk |
| `--output` | str | `output/output.mp4` | Output video file path |
| `--show-fps` | flag | `True` | Display FPS counter on screen |
| `--thickness` | int | `2` | Bounding box line thickness |
| `--imgsz` | int | `640` | Inference image size |

---

## 📁 Project Structure

```
real-time_face_detection_yolo/
├── detect_faces.py          # Main detection script
├── train.py                 # Training script for fine-tuning
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── LICENSE                  # MIT License
├── .gitignore              # Git ignore rules
├── data/
│   └── face_dataset.yaml   # Sample dataset configuration
├── utils/
│   ├── __init__.py         # Utils package init
│   └── visualization.py    # Drawing and FPS utilities
└── output/                 # Saved output videos (created automatically)
```

---

## 🔍 How It Works

### YOLOv26 Architecture

YOLOv26 is the latest evolution in the YOLO family, featuring:

- **NMS-Free Design**: Eliminates non-maximum suppression for faster inference
- **MuSGD Optimizer**: Advanced optimization for better convergence
- **Progressive Loss Balancing**: Automatic loss weight adjustment during training
- **Efficient Architecture**: Optimized backbone for real-time performance

### Detection Pipeline

1. **Video Input**: Capture frames from webcam or video file
2. **Preprocessing**: Resize and normalize frames for model input
3. **Inference**: Run YOLOv26 detection on each frame
4. **Post-processing**: Filter detections by confidence threshold
5. **Visualization**: Draw bounding boxes and labels
6. **Display**: Show annotated frames in real-time window

### Model Information

**Note**: The default pretrained YOLOv26 models are trained on the COCO dataset, which includes "person" detection but not specifically "face" detection. For best face detection results, you should fine-tune the model on a face-specific dataset like WIDER FACE.

---

## 🎓 Training on Custom Face Datasets

To train YOLOv26 for accurate face detection:

### 1. Prepare Your Dataset

Organize your face detection dataset in YOLO format:

```
datasets/face_detection/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val/
│       ├── img001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    └── val/
        ├── img001.txt
        └── ...
```

Each `.txt` file should contain normalized bounding boxes:
```
0 0.5 0.5 0.2 0.3
0 0.7 0.4 0.15 0.25
```

Format: `class_id x_center y_center width height` (all normalized to 0-1)

### 2. Update Dataset Configuration

Edit `data/face_dataset.yaml` to point to your dataset location.

### 3. Run Training

```bash
# Basic training
python train.py --data data/face_dataset.yaml --epochs 100

# Advanced training with custom parameters
python train.py --data data/face_dataset.yaml \
                --model yolo26m.pt \
                --epochs 200 \
                --batch 16 \
                --imgsz 640 \
                --name face_model_v1
```

### 4. Use Your Trained Model

```bash
python detect_faces.py --model runs/detect/face_model_v1/weights/best.pt
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | *required* | Path to dataset YAML file |
| `--model` | `yolo26n.pt` | Base model to fine-tune |
| `--epochs` | `100` | Number of training epochs |
| `--batch` | `16` | Batch size |
| `--imgsz` | `640` | Training image size |
| `--name` | `face_detection` | Experiment name |
| `--resume` | `None` | Resume from checkpoint |
| `--device` | auto | Device: `cuda:0`, `cpu`, or `mps` |
| `--workers` | `8` | Number of dataloader workers |
| `--patience` | `50` | Early stopping patience |

---

## 🐛 Troubleshooting

### Webcam Not Working

**Issue**: `Failed to open webcam (index 0)`

**Solutions**:
- Check if your webcam is connected and not being used by another application
- Try a different camera index: `--source 1` or `--source 2`
- On Linux, ensure you have permission to access `/dev/video0`
- Install proper camera drivers

### Model Download Fails

**Issue**: Cannot download YOLOv26 model weights

**Solutions**:
- Check your internet connection
- Manually download the model from [Ultralytics releases](https://github.com/ultralytics/assets/releases)
- Place the `.pt` file in the project directory

### Low FPS / Performance Issues

**Issue**: Detection is slow or laggy

**Solutions**:
- Use a smaller model: `--model yolo26n.pt` (nano is fastest)
- Lower the input resolution: `--imgsz 320`
- Ensure you have GPU support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Close other resource-intensive applications

### CUDA Out of Memory

**Issue**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `--batch 8` (for training)
- Use a smaller model variant
- Reduce image size: `--imgsz 320`
- Use CPU mode (slower): `--device cpu`

### No Detections Visible

**Issue**: Video shows but no bounding boxes appear

**Solutions**:
- Lower confidence threshold: `--conf 0.3` or `--conf 0.2`
- Ensure proper lighting and clear face visibility
- Use a face-tuned model for better results (see Training section)
- The default COCO model detects "person" - faces might be too small

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits & Acknowledgments

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** - YOLOv26 implementation and pretrained models
- **[OpenCV](https://opencv.org/)** - Computer vision library for video processing
- **YOLO Community** - For continuous innovations in real-time object detection

---

## 📞 Support & Contributing

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/kyroo404/real-time_face_detection_yolo/issues)
- **Pull Requests**: Contributions are welcome! Please open a PR with your improvements
- **Discussions**: Join the conversation in [GitHub Discussions](https://github.com/kyroo404/real-time_face_detection_yolo/discussions)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

**Made with ❤️ using YOLOv26**
