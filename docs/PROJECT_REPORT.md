
### 2. `docs/PROJECT_REPORT.md`

```markdown
# Robot Arm Color Sorting System - Project Report

## 1. Problem Definition

### 1.1 Problem Statement
Develop a computer vision system that can detect and classify colored objects in real-time, then simulate a robotic arm that automatically picks and sorts these objects based on their color.

### 1.2 Objectives
- Real-time detection of red, green, and blue objects
- Accurate color classification using HSV color space
- Stable object tracking with position stabilization
- Simulation of robotic pick-and-place operations
- Batch processing capabilities for multiple objects
- Comprehensive performance monitoring

### 1.3 Dataset
The system uses a custom color dataset (`colors.csv`) containing RGB values for various color shades, with specific focus on primary colors (Red, Green, Blue) for object classification.

## 2. Model Selection & Implementation

### 2.1 Computer Vision Approach

#### 2.1.1 Color Space Selection
**HSV (Hue, Saturation, Value)** was chosen over RGB for color detection because:
- Hue component provides consistent color representation
- Less sensitive to lighting variations
- Better separation of color information

#### 2.1.2 Detection Pipeline
1. **Frame Acquisition**: Capture video frames from webcam
2. **Preprocessing**: Gaussian blur for noise reduction
3. **HSV Conversion**: Convert BGR to HSV color space
4. **Color Segmentation**: Apply HSV range thresholds
5. **Morphological Operations**: Clean masks using opening/closing
6. **Contour Detection**: Find object boundaries
7. **Object Classification**: Verify colors and extract features

### 2.2 Key Implementation Details

#### 2.2.1 HSV Ranges
```python
HSV_RANGES = {
    "Red": [(0, 100, 100), (10, 255, 255)] + [(160, 100, 100), (180, 255, 255)],
    "Green": [(35, 50, 50), (90, 255, 255)],
    "Blue": [(95, 50, 50), (135, 255, 255)]
}
