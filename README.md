# SignBridge AI 

Real-time Sign Language Recognition System using Computer Vision and Deep Learning.

## Overview

SignBridge AI is an AI-powered sign language recognition system that detects and classifies hand gestures in real time using a webcam. The project uses MediaPipe for hand landmark detection and TensorFlow-based neural networks for gesture classification.

The system captures hand landmark coordinates, creates custom gesture datasets, trains deep learning models, and performs live gesture prediction with confidence scoring.

## Features

* Real-time hand tracking using MediaPipe
* Live gesture prediction through webcam
* Custom dataset collection pipeline
* Neural network-based gesture classification
* Real-time confidence display
* Scalable architecture for A-Z sign recognition
* Modular AI pipeline for future upgrades

## Tech Stack

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Pandas

## Current Progress

* Hand landmark detection
* Automatic dataset collection
* Neural network training pipeline
* Real-time gesture prediction

## Project Structure

```bash
SignBridge-AI/
│
├── dataset/
├── models/
├── src/
│   ├── collect_data.py
│   ├── inference.py
│   ├── predict.py
│   └── train_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

```text
Webcam
   ↓
MediaPipe Hand Tracking
   ↓
Landmark Extraction
   ↓
Neural Network Model
   ↓
Gesture Prediction
```

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/SignBridge-AI.git
cd SignBridge-AI
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

### Hand Tracking

```bash
python src/inference.py
```

### Dataset Collection

```bash
python src/collect_data.py
```

### Train Model

```bash
python src/train_model.py
```

### Real-Time Prediction

```bash
python src/predict.py
```

## Future Improvements

* Full A-Z gesture recognition
* Dynamic gesture recognition
* Sentence formation using NLP
* Text-to-speech conversion
* Raspberry Pi / Jetson deployment
* ESP32-based IoT interaction
* Mobile app integration
* Edge AI optimization

## Project Goal

The goal of SignBridge AI is to develop an intelligent assistive communication system that combines AI, Computer Vision, and Embedded Systems to improve accessibility and real-time human-computer interaction.

## Author

Developed as an AI + Embedded Systems engineering project focused on real-time computer vision and assistive AI technologies.
