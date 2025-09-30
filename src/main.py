import cv2
import numpy as np
import time
from color_detection import ColorDetector
from robot_arm_simulator import RobotArmSimulator
from object_tracker import ObjectSelector, PositionStabilizer

class RobotArmSortingSystem:
    def __init__(self, dataset_path):
        # Initialize components
        self.color_detector = ColorDetector(dataset_path)
        self.robot_sim = RobotArmSimulator()
        self.object_selector = ObjectSelector(tolerance_radius=40)
        self.position_stabilizer = PositionStabilizer()
        
        # Camera setup
        self.cap = None
        self.setup_camera()
        
        # Display settings
        self.setup_display()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def setup_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ Cannot open camera")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✅ Camera initialized successfully")
    
    def setup_display(self):
        """Setup display window"""
        cv2.namedWindow("Robot Arm Sorting System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Arm Sorting System", 1200, 700)
    
    def process_frame(self, frame):
        """Process single frame"""
        # Detect objects
        detected_objects = self.color_detector.detect_objects(frame)
        
        # Stabilize positions
        current_ids = []
        for obj in detected_objects:
            stable_pos = self.position_stabilizer.update(
                obj['id'], list(obj['pixel_pos'])
            )
            obj['pixel_pos'] = tuple(stable_pos)
            current_ids.append(obj['id'])
        
        # Cleanup old stabilizers
        self.position_stabilizer.cleanup(current_ids)
        
        return detected_objects
    
    def draw_objects(self, frame, objects):
        """Draw detected objects on frame"""
        for obj in objects:
            x, y, w, h = obj['bbox']
            center_x, center_y = obj['pixel_pos']
            color_name = obj['color']
            
            # Color mapping
            color_bgr = (0, 0, 255) if color_name == "Red" else (
                (0, 255, 0) if color_name == "Green" else (255, 0, 0)
            )
            
            # Check if selected
            is_selected = (not self.robot_sim.is_moving and 
                         self.object_selector.is_object_selected() and 
                         self.object_selector.selection_id == obj['id'])
            
            # Draw bounding box
            thickness = 4 if is_selected else 2
            box_color = (0, 255, 255) if is_selected else color_bgr
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, thickness)
            cv2.circle(frame, (center_x, center_y), 8, box_color, -1)
            
            # Draw tolerance circle for selected object
            if is_selected:
                cv2.circle(frame, (center_x, center_y), 
                          self.object_selector.tolerance_radius, (0, 255, 255), 2)
            
            # Label with coordinates
            robot_x, robot_y, robot_z = obj['robot_pos']
            label = f"{color_name} ({robot_x},{robot_y})"
            cv2.putText(frame, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        return frame
    
    def draw_drop_zones(self, frame):
        """Draw drop zones on frame"""
        height, width = frame.shape[:2]
        drop_zone_height = 80
        
        for color, (drop_x, drop_y, drop_z) in self.robot_sim.drop_locations.items():
            display_x = int(drop_x * width / 600)
            display_y = height - drop_zone_height + 20
            
            color_bgr = (0, 0, 255) if color == "Red" else (
                (0, 255, 0) if color == "Green" else (255, 0, 0)
            )
            
            cv2.rectangle(frame, (display_x-30, height-drop_zone_height), 
                         (display_x+30, height-10), color_bgr, 2)
            cv2.putText(frame, f"{color} Drop", (display_x-25, height-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
        
        return frame
    
    def create_history_panel(self, width, height):
        """Create history panel below camera feed"""
        history_panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(history_panel, "PICK HISTORY", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display recent history
        recent_history = self.robot_sim.pick_history[-4:]  # Last 4 picks
        
        if not recent_history:
            cv2.putText(history_panel, "No completed missions", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            # Display in columns
            col_width = width // 2
            for i, mission in enumerate(reversed(recent_history)):
                if i < 2:  # First column
                    x_pos = 20
                    y_pos = 60 + (i * 30)
                else:  # Second column
                    x_pos = col_width + 10
                    y_pos = 60 + ((i-2) * 30)
                
                time_str = time.strftime("%H:%M:%S", time.localtime(mission['timestamp']))
                color_bgr = (0, 0, 255) if mission['color'] == 'Red' else (
                    (0, 255, 0) if mission['color'] == 'Green' else (255, 0, 0)
                )
                drop_x, drop_y, _ = mission['drop_location']
                history_text = f"{time_str}: {mission['color']} -> ({drop_x},{drop_y})"
                
                cv2.putText(history_panel, history_text, (x_pos, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
        
        return history_panel
    
    def create_info_panel(self, objects, panel_height, panel_width):
        """Create information panel with proper layout"""
        info_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Add all information sections with proper spacing
        y_offset = self.add_header(info_panel)
        y_offset = self.add_status_section(info_panel, y_offset)
        y_offset = self.add_mission_progress(info_panel, y_offset)
        y_offset = self.add_controls_section(info_panel, y_offset)
        y_offset = self.add_selected_object(info_panel, y_offset)
        y_offset = self.add_detected_objects(info_panel, objects, y_offset)
        self.add_performance_metrics(info_panel, panel_height)
        
        return info_panel
    
    def add_header(self, panel):
        """Add header to info panel"""
        cv2.putText(panel, "ROBOT ARM SORTING SYSTEM", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return 50
    
    def add_status_section(self, panel, y_offset):
        """Add status section"""
        status_bg = (0, 100, 0) if not self.robot_sim.is_moving else (0, 0, 100)
        cv2.rectangle(panel, (5, y_offset), (panel.shape[1]-5, y_offset+50), status_bg, -1)
        
        status_text = "🤖 STATUS: READY" if not self.robot_sim.is_moving else "🤖 STATUS: WORKING"
        status_color = (0, 255, 0) if not self.robot_sim.is_moving else (0, 165, 255)
        
        cv2.putText(panel, status_text, (15, y_offset+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Wrap long status text
        status = self.robot_sim.get_status()
        if len(status) > 40:
            # Split status into two lines
            words = status.split()
            line1 = ""
            line2 = ""
            for word in words:
                if len(line1 + " " + word) <= 40:
                    line1 += " " + word
                else:
                    line2 += " " + word
            
            cv2.putText(panel, line1.strip(), (15, y_offset+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if line2:
                cv2.putText(panel, line2.strip(), (15, y_offset+55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                return y_offset + 70
        else:
            cv2.putText(panel, status, (15, y_offset+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return y_offset + 60
    
    def add_mission_progress(self, panel, y_offset):
        """Add mission progress section"""
        if self.robot_sim.is_moving:
            cv2.putText(panel, "MISSION PROGRESS:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            
            # Progress bar
            cv2.rectangle(panel, (10, y_offset), (panel.shape[1]-10, y_offset+20), (50, 50, 50), -1)
            progress_width = int((panel.shape[1]-20) * self.robot_sim.mission_progress / 100)
            cv2.rectangle(panel, (10, y_offset), (10+progress_width, y_offset+20), (0, 200, 0), -1)
            y_offset += 30
            
            current_step = self.robot_sim.get_current_step()
            if current_step:
                step_text = f"Step: {current_step['description']}"
                cv2.putText(panel, step_text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
            
            # Batch queue info
            if self.robot_sim.batch_queue:
                queue_text = f"Queue: {len(self.robot_sim.batch_queue)} objects remaining"
                cv2.putText(panel, queue_text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                y_offset += 20
        else:
            y_offset += 10
        
        return y_offset + 10
    
    def add_controls_section(self, panel, y_offset):
        cv2.putText(panel, "CONTROLS", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        # Controls in two columns
        controls_left = [
            "1-6: Select object",
            "SPACE: Pick selected",
            "F: Pick ALL objects",
            "R: Pick RED objects"
        ]
        
        controls_right = [
            "G: Pick GREEN objects",
            "B: Pick BLUE objects",
            "C: Clear selection",
            "+/-: Tolerance"
        ]
        
        # Left column
        for i, control in enumerate(controls_left):
            cv2.putText(panel, control, (15, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 18
        
        # Right column
        y_offset_right = y_offset - (len(controls_left) * 18)
        for i, control in enumerate(controls_right):
            cv2.putText(panel, control, (panel.shape[1]//2, y_offset_right), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset_right += 18
        
        return max(y_offset, y_offset_right) + 10
    
    def add_selected_object(self, panel, y_offset):
        if self.object_selector.is_object_selected() and not self.robot_sim.is_moving:
            cv2.putText(panel, "SELECTED OBJECT:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
            obj = self.object_selector.selected_object
            info_lines = [
                f"Color: {obj['color']}",
                f"Position: ({obj['robot_pos'][0]}, {obj['robot_pos'][1]})",
                f"Size: {obj['size'][0]}x{obj['size'][1]}",
                f"Area: {obj['area']:.0f} px"
            ]
            
            for line in info_lines:
                cv2.putText(panel, line, (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 18
            
            y_offset += 10
        else:
            cv2.putText(panel, "No object selected", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            y_offset += 40
        
        return y_offset
    
    def add_detected_objects(self, panel, objects, y_offset):
        cv2.putText(panel, "DETECTED OBJECTS:", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        if not objects:
            cv2.putText(panel, "No objects detected", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            return y_offset + 30
        
        # Color counts
        red_count = len([obj for obj in objects if obj['color'] == 'Red'])
        green_count = len([obj for obj in objects if obj['color'] == 'Green'])
        blue_count = len([obj for obj in objects if obj['color'] == 'Blue'])
        
        cv2.putText(panel, f"Red: {red_count}  Green: {green_count}  Blue: {blue_count}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # List objects (show max 6)
        max_objects = min(6, len(objects))
        for i in range(max_objects):
            obj = objects[i]
            color_bgr = (0, 0, 255) if obj['color'] == 'Red' else (
                (0, 255, 0) if obj['color'] == 'Green' else (255, 0, 0)
            )
            
            is_selected = (self.object_selector.is_object_selected() and 
                          self.object_selector.selection_id == obj['id'])
            prefix = "▶ " if is_selected else f"{i+1}. "
            
            obj_text = f"{prefix}{obj['color']} - ({obj['robot_pos'][0]},{obj['robot_pos'][1]})"
            cv2.putText(panel, obj_text, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 2 if is_selected else 1)
            y_offset += 18
        
        # Show "more objects" indicator if needed
        if len(objects) > max_objects:
            remaining = len(objects) - max_objects
            cv2.putText(panel, f"... and {remaining} more", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
        
        return y_offset + 10
    
    def add_performance_metrics(self, panel, panel_height):
        """Add performance metrics at bottom"""
        metrics = self.robot_sim.get_performance_metrics()
        
        metrics_text = f"Picks: {metrics['total_picks']} | Success: {metrics['success_rate']:.1f}%"
        cv2.putText(panel, metrics_text, (10, panel_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Tolerance info
        tolerance_text = f"Tolerance: {self.object_selector.tolerance_radius}px"
        text_size = cv2.getTextSize(tolerance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(panel, tolerance_text, (panel.shape[1] - text_size[0] - 10, panel_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def handle_keyboard_input(self, key, objects):
        """Handle keyboard input"""
        if key == ord('q'):
            return False
        elif key == ord('c'):
            self.object_selector.clear_selection()
            print("Selection cleared")
        elif key == ord('+'):
            self.object_selector.tolerance_radius = min(100, self.object_selector.tolerance_radius + 5)
            print(f"Tolerance increased to {self.object_selector.tolerance_radius}px")
        elif key == ord('-'):
            self.object_selector.tolerance_radius = max(10, self.object_selector.tolerance_radius - 5)
            print(f"Tolerance decreased to {self.object_selector.tolerance_radius}px")
        elif key == ord(' ') and not self.robot_sim.is_moving:
            if self.object_selector.is_object_selected():
                obj = self.object_selector.selected_object
                self.robot_sim.start_mission(obj['robot_pos'], obj['color'])
                self.object_selector.clear_selection()
                print(f"Started mission for {obj['color']} object")
        elif key == ord('f') and not self.robot_sim.is_moving:
            self.robot_sim.start_all_objects_batch(objects)
            self.object_selector.clear_selection()
            print("Started batch pick for ALL objects")
        elif key in [ord('r'), ord('g'), ord('b')] and not self.robot_sim.is_moving:
            color_map = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}
            color = color_map[chr(key)]
            self.robot_sim.start_color_batch_pick(objects, color)
            self.object_selector.clear_selection()
            print(f"Started batch pick for {color} objects")
        elif key in [ord(str(i)) for i in range(1, 7)] and not self.robot_sim.is_moving:
            obj_index = int(chr(key)) - 1
            if obj_index < len(objects):
                self.object_selector.select_object(objects[obj_index])
                print(f"Selected {objects[obj_index]['color']} object")
        
        return True
    
    def calculate_fps(self):
        """Calculate frames per second"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - self.start_time)
            self.start_time = end_time
            return fps
        return None
    
    def run(self):
        """Main system loop"""
        print("🚀 Starting Robot Arm Sorting System...")
        print("Controls: 1-6=Select, SPACE=Pick, F=All, R/G/B=Color, C=Clear, Q=Quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Process frame
            objects = self.process_frame(frame)
            
            # Update robot mission
            self.robot_sim.update_mission()
            
            # Update object selection
            if self.robot_sim.is_moving:
                self.object_selector.clear_selection()
            else:
                self.object_selector.update_selection(objects)
            
            # Draw objects and UI
            frame = self.draw_objects(frame, objects)
            frame = self.draw_drop_zones(frame)
            
            # Create main display (camera feed)
            main_display_height = 400
            main_display_width = int(frame.shape[1] * main_display_height / frame.shape[0])
            main_display = cv2.resize(frame, (main_display_width, main_display_height))
            
            # Create history panel
            history_panel = self.create_history_panel(main_display_width, 100)
            
            # Combine camera and history
            left_display = np.vstack([main_display, history_panel])
            
            # Create info panel with matching height
            info_panel_height = left_display.shape[0]
            info_panel_width = 450
            info_panel = self.create_info_panel(objects, info_panel_height, info_panel_width)
            
            # Combine all displays
            final_display = np.hstack([left_display, info_panel])
            
            # Add FPS
            fps = self.calculate_fps()
            if fps:
                cv2.putText(final_display, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Robot Arm Sorting System", final_display)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_keyboard_input(key, objects):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ System shutdown complete")

def main():
    try:
        dataset_path = "data/colors.csv"
        system = RobotArmSortingSystem(dataset_path)
        system.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the camera is connected and colors.csv exists in data folder")

if __name__ == "__main__":
    main()