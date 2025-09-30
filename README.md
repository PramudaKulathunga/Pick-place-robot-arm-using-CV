# 🤖 Robot Arm Color Sorting System

A computer vision-based system for detecting colored objects and simulating robotic arm pick-and-place operations.

## 🎯 Project Overview

This system uses OpenCV for real-time color detection and classification of objects, then simulates a robotic arm that picks detected objects and sorts them into designated drop zones based on their color.

## ✨ Features

- **Real-time Color Detection**: Red, Green, Blue object detection using HSV color space
- **Object Tracking**: Stable selection with tolerance-based tracking
- **Robot Arm Simulation**: Complete pick-and-place mission simulation
- **Batch Operations**: Pick all objects or specific colors automatically
- **Performance Metrics**: Track success rates and mission statistics
- **Interactive GUI**: Real-time visualization with comprehensive controls

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Webcam
- OpenCV

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/robot-arm-color-sorting.git
cd robot-arm-color-sorting

# Install dependencies
pip install -r requirements.txt

# Run the system
python run_system.py
