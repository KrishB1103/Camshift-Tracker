# Camshift-Tracker
# Real-Time Object Tracking using CAMShift + Kalman Filter (OpenCV)

This project implements a real-time object tracking system using classical Computer Vision techniques instead of deep learning.
The tracker learns the color distribution of a user-selected object and continuously follows it in a live webcam feed.

To improve stability and realism, a Kalman Filter is integrated to predict motion and smooth noisy detections.

---

## Features

* Interactive object selection using mouse
* Color-based tracking using HSV histogram
* Back Projection probability mapping
* Adaptive tracking window using CAMShift
* Motion prediction using Kalman Filter
* Smooth tracking even during fast motion or temporary occlusion
* Real-time performance on CPU (no GPU required)

---

## How it Works

### 1. Object Learning

The user selects an object in the first frame.
The program computes an HSV color histogram representing the object's appearance.

### 2. Detection (CAMShift)

For every new frame:

* The histogram is projected onto the image (Back Projection)
* CAMShift finds the most probable region
* The tracking window adapts size and rotation

### 3. Motion Prediction (Kalman Filter)

The Kalman Filter estimates:

* Position
* Velocity
* Future location

This reduces jitter and allows the tracker to continue during short occlusions.

---

## Pipeline

Camera Frame → HSV Conversion → Back Projection → CAMShift Detection → Kalman Prediction → Smoothed Tracking Output

---

## Tech Stack

* Python
* OpenCV
* NumPy

---

## Why this project?

Most modern trackers rely on deep learning models like YOLO + SORT.
This project demonstrates the mathematical foundations behind tracking systems, including probability modeling, optimization, and state estimation.

It helps understand how tracking works internally before using neural networks.

---

## Future Improvements

* Multi-object tracking
* DeepSORT integration
* Re-identification after long occlusion
* Deep learning detector (YOLO) + Kalman tracking

---

## Demo

Select any colored object → Press ENTER → Move object
Blue box: detection (CAMShift)
Red dot: predicted position (Kalman Filter)
