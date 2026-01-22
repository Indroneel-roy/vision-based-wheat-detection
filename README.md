# 🌾 Wheat Head Detection using YOLOv11

An AI-powered system that automatically detects and counts wheat heads in images using YOLOv11 deep learning model.

![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)(https://huggingface.co/spaces/Indroneel/wheat-detection)

## 🎯 What is This?

This project helps farmers and researchers automatically count wheat heads in field images. Just upload a photo and get instant results with bounding boxes around each wheat head.

**Try it live:** [Wheat Detection App](https://huggingface.co/spaces/Indroneel/wheat-detection)

## ✨ Key Features

- 🚀 Fast and accurate detection using YOLOv11
- 🌐 Easy-to-use web interface
- 📊 Download results as images or JSON
- ⚡ GPU-accelerated inference
- 🎯 Adjustable confidence threshold

## 📊 Model Performance

Trained on **3,422 images** from the Global Wheat Detection dataset:

- **mAP@50**: 94.7%
- **Precision**: 91.7%
- **Recall**: 89.2%
- **Training Time**: 1.7 hours (Tesla T4 GPU)

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Indroneel-roy/vision-based-wheat-detection.git
cd vision-based-wheat-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Web App

```bash
streamlit run frontend/app.py
```

Open your browser at `http://localhost:8501`

## 📁 Project Structure

```
vision-based-wheat-detection/
│
├── backend/              # FastAPI backend server
├── frontend/             # Streamlit web interface  
├── notebooks/            # Training notebooks
│   └── object_detection_yolov11m.py
├── requirements.txt      # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

## 🏋️ Training Your Own Model

The training notebook includes:

1. **Data Exploration** - Visualize dataset statistics
2. **Preprocessing** - Convert to YOLO format
3. **Training** - Fine-tune YOLOv11 model
4. **Evaluation** - Test and validate results

**Dataset**: [Global Wheat Detection (Kaggle)](https://www.kaggle.com/c/global-wheat-detection)

**Training Configuration**:
- Model: YOLOv11-Medium
- Epochs: 50
- Image Size: 640×640
- Batch Size: 8
- GPU: Tesla T4

## 💻 Usage Examples

### Detect in Python

```python
from ultralytics import YOLO

# Load model
model = YOLO("path/to/best.pt")

# Run detection
results = model.predict(
    source="wheat_image.jpg",
    conf=0.25,
    save=True
)

# Get detections
for result in results:
    boxes = result.boxes
    print(f"Found {len(boxes)} wheat heads")
```

### Web Interface Features

- Upload images (JPG, PNG)
- Adjust confidence threshold (0.0 - 1.0)
- View detection statistics
- Download annotated images
- Export results as JSON

## 🛠️ Technologies

- **YOLOv11** - Object detection model
- **FastAPI** - Backend API
- **Streamlit** - Web interface
- **PyTorch** - Deep learning framework
- **OpenCV** - Image processing

## 📈 Results

The model successfully detects wheat heads with:
- High accuracy across different lighting conditions
- Robust to various wheat varieties
- Fast inference (~30-50ms per image)

## 🤝 Contributing

Contributions welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [Global Wheat Detection Challenge](https://www.kaggle.com/c/global-wheat-detection) - Dataset
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv11 implementation
- [Hugging Face](https://huggingface.co/) - Deployment platform

## 📧 Contact

**Indroneel Roy**
- GitHub: [@Indroneel-roy](https://github.com/Indroneel-roy)
- Hugging Face: [@Indroneel](https://huggingface.co/Indroneel)

---

⭐ **Star this repo if you find it helpful!** ⭐
