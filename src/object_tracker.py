import numpy as np
import math
from collections import deque

class ObjectSelector:

    def __init__(self, tolerance_radius=30):
        self.tolerance_radius = tolerance_radius
        self.selected_object = None
        self.selection_id = None
        
    def update_selection(self, current_objects):
        """Update selection based on tolerance range"""
        if self.selected_object is None:
            return None
        
        selected_x, selected_y = self.selected_object['pixel_pos']
        
        # Find objects within tolerance radius
        matching_objects = []
        for obj in current_objects:
            current_x, current_y = obj['pixel_pos']
            distance = math.sqrt((current_x - selected_x) ** 2 + (current_y - selected_y) ** 2)
            
            if distance <= self.tolerance_radius:
                matching_objects.append((obj, distance))
        
        if matching_objects:
            # Select the closest matching object
            matching_objects.sort(key=lambda x: x[1])
            closest_obj = matching_objects[0][0]
            
            # Update selection with current object data
            self.selected_object = closest_obj
            self.selection_id = closest_obj['id']
            return closest_obj
        else:
            return None
    
    def select_object(self, obj):
        """Select a new object"""
        self.selected_object = obj
        self.selection_id = obj['id']
        return obj
    
    def clear_selection(self):
        """Clear current selection"""
        self.selected_object = None
        self.selection_id = None
    
    def is_object_selected(self):
        """Check if any object is selected"""
        return self.selected_object is not None


class PositionStabilizer:

    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.position_buffers = {}
        
    def update(self, obj_id, position):
        """Update position with stabilization"""
        if obj_id not in self.position_buffers:
            self.position_buffers[obj_id] = deque(maxlen=self.buffer_size)
        
        self.position_buffers[obj_id].append(position)
        
        if len(self.position_buffers[obj_id]) == self.buffer_size:
            return np.median(self.position_buffers[obj_id], axis=0).astype(int)
        return position
    
    def cleanup(self, current_ids):
        """Remove old stabilizers"""
        self.position_buffers = {k: v for k, v in self.position_buffers.items() if k in current_ids}