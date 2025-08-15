### Real-time Face AI Suite


## Overview
This project is a comprehensive real-time face analysis application built using OpenCV, Dlib, Keras, and face_recognition. It goes beyond basic face detection to offer a wide array of features including face tracking, age and gender estimation, emotion recognition, face identification, drowsiness and yawning detection, smiling detection, and dynamic visual filters based on detected emotions. It runs on a live webcam feed, providing an interactive experience with detailed face analytics.

## Features
- **Real-time Face Detection:** Uses a Caffe-based SSD (Single Shot Detector) model for robust detection.
- **Persistent Face Tracking:** Maintains tracking of detected faces across frames using CSRT algorithm.
- **Age and Gender Estimation:** Predicts age range and gender with pre-trained deep learning models.
- **Emotion Recognition:** Classifies facial expressions into Happy, Sad, Angry, Neutral, Surprise, Fear, and Disgust.
- **Face Recognition/Identification:** Identifies known individuals; unknown faces are flagged and saved.
- **Drowsiness Detection:** Monitors eye aspect ratio (EAR) to detect drowsiness.
- **Yawning Detection:** Detects yawning using mouth aspect ratio (MAR).
- **Smiling Detection:** Determines if a person is smiling.
- **Head Pose Estimation:** Estimates the 3D orientation of the head.
- **Dynamic Visual Filters:** Applies fun overlays based on detected emotions.
- **Face Morphing Effects:** Applies cartoonify and aging filters based on specific emotions.
- **Performance Metrics:** Displays real-time FPS and average processing times.
- **Emotion Heatmap:** Visualizes dominant emotions with color-coded bounding boxes.
- **Analytics Panel:** Summarizes tracked faces and counts each detected emotion.

---
## Directory Structure
```bash
├── deploy/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── models/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   ├── fer2013_mini_XCEPTION.102-0.66.hdf5
│   └── shape_predictor_68_face_landmarks.dat
├── filters/
│   ├── sunglasses.png
│   ├── hat.png
│   └── angry_emoji.png
├── known_faces/
│   ├── your_name.jpg
│   └── another_person.png
├── outputs/
├── unknown_faces/
├── main.py  
└── README.md
└── requirements.txt


```
---


## Prerequisites
- Python 3.x
- Libraries:
  - opencv-python
  - numpy
  - dlib
  - face_recognition
  - keras (or tensorflow)
  - scipy

## Installation
1. Clone the repository:
```bash
git clone <https://github.com/arpanneupane75/AI-faceSuite.git>
cd Ultimate-Face-AI-Suit
```
2. create the necessary directories
```bash 
mkdir deploy models filters known_faces outputs unknown_faces
```
3.Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install opencv-python numpy dlib face_recognition keras scipy
```

## Model Setup
This project relies on several pre-trained deep learning models. You need to download them and place them in the correct directories as specified in the `Directory Structure` section.

*   **Face Detection Model (Caffe SSD):**
    *   `deploy.prototxt`
    *   `res10_300x300_ssd_iter_140000.caffemodel`
    *   Download from: [OpenCV DNN Face Detector Links](#1-face-detection-model-caffe-ssd)
    *   Place both files in the `deploy/` directory.

*   **Age and Gender Estimation Models (Caffe):**
    *   `age_deploy.prototxt`
    *   `age_net.caffemodel`
    *   `gender_deploy.prototxt`
    *   `gender_net.caffemodel`
    *   Download from: [Age and Gender Classification Links](#2-age-and-gender-estimation-models-caffe)
    *   Place all four files in the `models/` directory.

*   **Emotion Recognition Model (Keras HDF5):**
    *   `fer2013_mini_XCEPTION.102-0.66.hdf5`
    *   Download from: [Emotion Recognition GitHub Links](#3-emotion-recognition-model-keras-hdf5) (or similar sources for a FER2013-trained model)
    *   Place the file in the `models/` directory.

*   **Dlib Shape Predictor:**
    *   `shape_predictor_68_face_landmarks.dat`
    *   Download from: [Dlib Models Link](#4-dlib-shape-predictor) (You'll need to decompress the `.bz2` file)
    *   Place the file in the `models/` directory.

## Known Faces Setup
To enable face recognition:

1.  Place images of known individuals in the `known_faces/` directory.
2.  Name each image file with the person's name (e.g., `John_Doe.jpg`, `Jane_Smith.png`). The script will use the filename (without extension) as the person's name.
3.  Ensure each image contains a clear, frontal view of the person's face.

## Filters Setup
The project uses sample filter images. You can customize these:

1.  The `filters/` directory contains `sunglasses.png`, `hat.png`, and `angry_emoji.png`.
2.  These are loaded based on emotion labels. You can replace them with your own images (PNGs with transparency are best) or add more, but you'll need to update the `filter_paths` dictionary in the `main.py` script to link them to specific emotions.

## Usage

1.  Ensure all prerequisites are met and models are correctly placed.
2.  Open a terminal or command prompt.
3.  Navigate to the project's root directory.
4.  Activate your virtual environment (if you created one):
    ```bash
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```
5.  Run the main script:
    ```bash
    python main.py
    ```
6.  A window will open displaying your webcam feed with the applied features.
7.  Press `q` to quit the application.

## Troubleshooting

*   **"Error: Could not open webcam."**: Ensure your webcam is connected, drivers are updated, and no other application is currently using it.
*   **"Error loading [Model Name]..."**: Double-check that all model files are downloaded correctly and placed in their respective `deploy/` or `models/` directories. Also, verify file names are exact.
*   **`dlib` installation issues**: Refer to `dlib`'s official documentation or online resources for system-specific installation guides (e.g., installing CMake, Visual C++ Build Tools for Windows, or `build-essential` for Linux).
*   **"No known faces found..."**: If this warning appears, ensure you have images in the `known_faces` folder.
*   **Performance is slow**: Face analysis (especially emotion and age/gender) is computationally intensive. The `FRAME_SKIP` constant can be adjusted to process these features less frequently, improving FPS at the cost of less real-time updates for those specific features. Ensure you are not running many other demanding applications.

## Contributing

Feel free to fork the repository, open issues, or submit pull requests to improve this project.

## License

MIT License# AI-faceSuite
