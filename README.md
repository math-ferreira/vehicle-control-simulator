# Vehicle Control Simulator

This repository contains a modular system for recognizing and simulating vehicle control actions using computer vision and machine learning. The project is organized into several components, each responsible for detecting and interpreting different driver actions, such as steering, gear shifting, pedal operation, and hand/head movements.

## Project Structure

- **gear/**
  - `gear-shift-recognition.py`: Recognizes gear shift actions.
- **hand/**
  - `hand_landmarker.task`: Model/task file for hand landmark detection.
  - `hands-recognition.py`: Recognizes hand gestures or positions.
- **head/**
  - `head-recognition.py`: Recognizes head movements or orientation.
- **pedals/**
  - `pedal-recognition-v2.py`, `pedals-recognition.py`: Recognize pedal actions (accelerator, brake, clutch).
- **steering-wheel/**
  - `steering-wheel-recognition.py`: Recognizes steering wheel movements.
- **.task files**
  - `hand_landmarker.task`, `pose_landmarker_full.task`, `pose_landmarker_lite.task`: Pre-trained models or task files for landmark detection.

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
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   (Dependencies will depend on the recognition scripts, e.g., OpenCV, MediaPipe, etc. Add a `requirements.txt` if available.)
   ```bash
   pip install -r requirements.txt
   ```
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
*Edit this README to add more details about usage, dependencies, and contribution guidelines as your project evolves.*
