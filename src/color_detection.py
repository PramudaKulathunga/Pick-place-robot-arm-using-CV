import cv2
import numpy as np
import pandas as pd
from collections import deque


class ColorDetector:

    def __init__(self, dataset_path):
        """
        Initialize color detector with dataset
        """
        self.load_color_dataset(dataset_path)
        self.setup_hsv_ranges()
        
    def load_color_dataset(self, dataset_path):
        """
        Load color dataset from CSV file
        """
        try:
            df = pd.read_csv(dataset_path)
            self.color_names = df["Air Force Blue (Raf)"].tolist()
            self.R = df["93"].tolist()
            self.G = df["138"].tolist()
            self.B = df["168"].tolist()
            print(f"✅ Loaded {len(self.color_names)} colors from dataset")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            # Fallback to default colors
            self.setup_default_colors()
    
    def setup_default_colors(self):
        """
        Setup default color ranges if dataset fails to load
        """
        self.color_names = ["Red", "Green", "Blue"]
        self.R = [255, 0, 0]
        self.G = [0, 255, 0]
        self.B = [0, 0, 255]
        print("⚠️  Using default color ranges")
    
    def setup_hsv_ranges(self):
        """
        Define HSV ranges for color detection
        """
        self.hsv_ranges = {
            "Red": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            "Green": [
                (np.array([35, 50, 50]), np.array([90, 255, 255]))
            ],
            "Blue": [
                (np.array([95, 50, 50]), np.array([135, 255, 255]))
            ]
        }
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for better color detection
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def create_color_mask(self, hsv_frame, color_name):
        """
        Create mask for specific color
        """
        if color_name not in self.hsv_ranges:
            return None
            
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.hsv_ranges[color_name]:
            color_mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
        
        return self.clean_mask(mask)
    
    def clean_mask(self, mask):
        """
        Apply morphological operations to clean the mask
        """
        kernel = np.ones((5, 5), np.uint8)
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Enhance object shapes
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def detect_objects(self, frame, min_area=1000):
        """
        Detect colored objects in the frame
        """
        hsv_frame = self.preprocess_frame(frame)
        detected_objects = []
        
        for color_name in self.hsv_ranges.keys():
            mask = self.create_color_mask(hsv_frame, color_name)
            
            if mask is not None:
                contours = self.find_contours(mask, min_area)
                objects = self.process_contours(contours, color_name, frame.shape)
                detected_objects.extend(objects)
        
        return detected_objects
    
    def find_contours(self, mask, min_area):
        """
        Find contours in the mask
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Check contour solidity
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                if solidity > 0.7:  # Only accept fairly convex shapes
                    filtered_contours.append(cnt)
        
        return filtered_contours
    
    def process_contours(self, contours, color_name, frame_shape):
        """
        Process contours into object information
        """
        objects = []
        height, width = frame_shape[:2]
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Convert to robot coordinates
            robot_x = int(center_x * 600 / width)
            robot_y = int(center_y * 400 / height)
            robot_z = 20  # Object height
            
            obj_info = {
                'pixel_pos': (center_x, center_y),
                'robot_pos': (robot_x, robot_y, robot_z),
                'size': (w, h),
                'area': cv2.contourArea(cnt),
                'color': color_name,
                'bbox': (x, y, w, h),
                'id': f"{color_name}_{center_x}_{center_y}"
            }
            
            objects.append(obj_info)
        
        return objects
    
    def verify_color(self, frame, bbox, expected_color):
        """
        Verify detected color by sampling from object region
        """
        x, y, w, h = bbox
        
        # Sample from center region to avoid edges
        roi = frame[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4]
        
        if roi.size > 0:
            avg_b = np.median(roi[:,:, 0])
            avg_g = np.median(roi[:,:, 1])
            avg_r = np.median(roi[:,:, 2])
            
            detected_color = self.get_color_name_rgb(int(avg_r), int(avg_g), int(avg_b))
            return detected_color == expected_color
        
        return False
    
    def get_color_name_rgb(self, r, g, b):
        """
        Get color name from RGB values using HSV conversion
        """
        pixel = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0][0]
        
        if s > 50 and v > 50:
            if (0 <= h <= 10) or (170 <= h <= 180):
                return "Red"
            elif 35 <= h <= 85:
                return "Green"
            elif 100 <= h <= 130:
                return "Blue"
        
        return "Other"
