
# Vehicle Control Simulator

This repository provides a modular system for recognizing and simulating vehicle control actions using computer vision and machine learning. Each component detects and interprets different driver actions, such as steering, gear shifting, pedal operation, and hand/head movements.


## Project Structure

- **gear/**
  - `gear-shift-recognition.py`: Recognizes gear shift actions.
- **hand/**
  - `hand_landmarker.task`: Model/task file for hand landmark detection.
  - `hands-recognition.py`: Recognizes hand gestures or positions.
- **head/**
  - `head-recognition.py`: Recognizes head movements or orientation.
- **pedals/**
  - `pedals-recognition-v2.py`, `pedals-recognition.py`: Recognize pedal actions (accelerator, brake, clutch).
- **steering-wheel/**
  - `hand_landmarker.task`: Model/task file for hand landmark detection (specific to steering wheel).
  - `steering-wheel-recognition-v2.py`, `steering-wheel-recognition.py`: Recognize steering wheel movements.
- **Root directory**
  - `camera-recognition.py`, `open-camera.py`: General camera and recognition utilities.
  - `hand_landmarker.task`, `pose_landmarker_full.task`, `pose_landmarker_lite.task`: Pre-trained models for landmark detection.


## Features

- Modular recognition scripts for each vehicle control (steering, pedals, gear, hand, head)
- Uses machine learning models for landmark and pose detection
- Designed for integration with simulators or real-time feedback systems


## Getting Started

1. **Clone the repository:**
  ```bash
  git clone <repo-url>
  cd vehicle-control-simulator
  ```
2. **Set up a Python virtual environment (recommended):**
  ```bash
  python -m venv venv
  # On Windows:
  venv\Scripts\activate
  # On macOS/Linux:
  source venv/bin/activate
  ```
3. **Install dependencies:**
  - If a `requirements.txt` is available:
    ```bash
    pip install -r requirements.txt
    ```
  - Otherwise, install dependencies as required by each script (e.g., OpenCV, MediaPipe).
4. **Run a recognition module:**
  ```bash
  python steering-wheel/steering-wheel-recognition.py
  ```
  Replace with the desired module/script.


## Requirements

- Python 3.8+
- OpenCV, MediaPipe, and other dependencies as required by each script
- Webcam or video input for real-time recognition


## Usage

Each module can be run independently to recognize specific vehicle control actions. The scripts are designed to be extensible and can be integrated into larger simulation or driver monitoring systems.


## License

Specify your license here (e.g., MIT, Apache 2.0, etc.)


## Acknowledgments

- MediaPipe for landmark detection models
- OpenCV for image processing

---

*Update this README as your project evolves, especially if you add new modules, dependencies, or usage instructions.*

---
*Edit this README to add more details about usage, dependencies, and contribution guidelines as your project evolves.*
