# Hand Gesture Recognition

A real-time hand gesture recognition system using MediaPipe landmarks and machine learning. This project detects hand gestures from webcam input using trained classification models.

## Overview

This project implements an end-to-end hand gesture recognition pipeline:
- Extracts hand landmarks using MediaPipe's pre-trained hand detector
- Trains multiple machine learning models (Random Forest, SVM, XGBoost)
- Performs real-time gesture classification from live webcam feed
- Visualizes predictions with landmarks and confidence scores

## Features

- **Landmark-based Classification**: Uses 21 hand landmarks with 3D coordinates (63 features)
- **Multi-Model Training**: Trains and compares Random Forest, SVM (RBF kernel), and XGBoost
- **Real-Time Inference**: Live webcam gesture prediction with visual feedback
- **Visualization**: Overlays hand landmarks, predicted gesture label, and confidence on video feed
- **Output Recording**: Saves annotated video and individual frames as images


## Prerequisites

- Python 3.8 or higher
- Webcam access (for real-time inference)
- 4GB+ RAM recommended

## Installation

1. **Clone or download the project**

2. **Create and activate a virtual environment** (recommended):
```bash
python -m venv gesture_venv
gesture_venv\Scripts\activate  # Windows
# or
source gesture_venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training Models

Open and execute the Jupyter notebook:
```bash
jupyter notebook Hand_Gestures.ipynb
```

The notebook performs the following steps:
1. Load hand-landmark dataset from CSV
2. Normalize landmarks (centered on wrist, scaled by middle fingertip distance)
3. Encode gesture labels using `LabelEncoder`
4. Split data into training and test sets
5. Train and evaluate multiple models
6. Save the best model (SVM) and label encoder using joblib

### Real-Time Inference

Run the webcam script:
```bash
python hand_gesture_camera.py
```

**Controls**:
- `q` - Stop recording and exit
- `s` - Capture current frame as screenshot

## Configuration

Before running inference, ensure these files are in the project directory:
- `svm_hand_gesture_model.pkl` - Trained SVM model
- `label_encoder.pkl` - Label encoder for decoding predictions
- `hand_landmarker.task` - MediaPipe hand landmark model (optional; auto-downloads if missing)

## Output Files

- `output_gesture_recognition.mp4` - Annotated video with gesture predictions
- `screenshot_YYYYMMDD_HHMMSS.jpg` - Individual frame captures

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not found | Verify webcam is connected and not in use by other applications |
| Model file not found | Ensure pickle files are in the project root directory |
| Slow performance | Check system resources; close unnecessary applications |
| ImportError for dependencies | Verify virtual environment is activated and all packages installed |

## Technical Details

- **Hand Detection**: MediaPipe BlazePalm
- **Feature Extraction**: 21 landmarks × 3 coordinates (x, y, z)
- **Normalization**: Wrist-centered, scale-invariant representation
- **Primary Model**: Support Vector Machine (RBF kernel)
- **Alternative Models**: Random Forest, XGBoost (for comparison)

## License

This project is provided as-is for educational purposes.

## Author

Developed as part of ITI Supervised Learning Course
