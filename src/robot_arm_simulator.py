import time
import numpy as np

class RobotArmSimulator:

    def __init__(self):
        self.workspace_width = 600
        self.workspace_height = 400
        
        # Drop locations for different colors
        self.drop_locations = {
            "Red": (100, 350, 50),
            "Green": (300, 350, 50),
            "Blue": (500, 350, 50)
        }
        
        # Robot state
        self.current_position = (300, 200, 100)
        self.gripper_state = "open"
        self.is_moving = False
        self.current_mission = None
        self.mission_progress = 0
        self.mission_steps = []
        self.batch_mode = False
        self.batch_queue = []
        self.pick_history = []
        
        # Performance metrics
        self.metrics = {
            "total_picks": 0,
            "successful_picks": 0,
            "failed_picks": 0,
            "average_pick_time": 0,
            "batch_completions": 0
        }
    
    def calculate_pick_mission(self, object_pos, object_color):
        """Calculate complete pick and place mission sequence"""
        object_x, object_y, object_z = object_pos
        drop_x, drop_y, drop_z = self.drop_locations[object_color]
        
        mission = [
            {"type": "move", "target": (object_x, object_y, object_z + 50), "description": "Approach object"},
            {"type": "move", "target": (object_x, object_y, object_z), "description": "Lower to object"},
            {"type": "gripper", "action": "close", "description": "Pick up object"},
            {"type": "move", "target": (object_x, object_y, object_z + 80), "description": "Lift object"},
            {"type": "move", "target": (drop_x, drop_y, object_z + 80), "description": "Move above drop zone"},
            {"type": "move", "target": (drop_x, drop_y, drop_z), "description": "Lower to drop height"},
            {"type": "gripper", "action": "open", "description": "Release object"},
            {"type": "move", "target": (drop_x, drop_y, drop_z + 80), "description": "Lift from drop zone"},
            {"type": "move", "target": self.current_position, "description": "Return to home"}
        ]
        
        return mission
    
    def update_mission(self):
        """Update mission progress and handle batch operations"""
        if self.is_moving and self.current_mission:
            self.mission_progress += 2
            
            if self.mission_progress >= 100:
                self.complete_mission()
    
    def complete_mission(self):
        """Complete current mission and handle next in batch"""
        self.mission_progress = 0
        
        # Record successful pick
        self.metrics["successful_picks"] += 1
        self.metrics["total_picks"] += 1
        
        # Add to history
        self.pick_history.append({
            "color": self.current_mission['object_color'],
            "position": self.current_mission['object_pos'],
            "timestamp": time.time(),
            "drop_location": self.drop_locations[self.current_mission['object_color']],
            "success": True
        })
        
        print(f"✅ MISSION COMPLETED: {self.current_mission['object_color']} object placed")
        
        # Handle batch queue
        if self.batch_queue:
            self.start_next_mission()
        else:
            if self.batch_mode:
                self.metrics["batch_completions"] += 1
                print("🎉 BATCH COMPLETED!")
            self.current_mission = None
            self.is_moving = False
            self.batch_mode = False
    
    def start_next_mission(self):
        """Start next mission in batch queue"""
        next_mission = self.batch_queue.pop(0)
        self.current_mission = {
            "object_pos": next_mission['robot_pos'],
            "object_color": next_mission['color'],
            "start_time": time.time()
        }
        self.mission_steps = self.calculate_pick_mission(
            next_mission['robot_pos'],
            next_mission['color']
        )
        print(f"🔄 STARTING NEXT: {next_mission['color']} object ({len(self.batch_queue)} remaining)")
    
    def start_mission(self, object_pos, object_color):
        """Start a new pick and place mission"""
        if not self.is_moving:
            self.current_mission = {
                "object_pos": object_pos,
                "object_color": object_color,
                "start_time": time.time()
            }
            self.mission_steps = self.calculate_pick_mission(object_pos, object_color)
            self.is_moving = True
            self.mission_progress = 0
            
            print(f"🤖 STARTING MISSION: Pick {object_color} object")
    
    def start_batch_pick(self, objects_list):
        """Start batch picking for multiple objects"""
        if not self.is_moving and objects_list:
            self.batch_mode = True
            self.batch_queue = objects_list[1:]
            first_object = objects_list[0]
            
            self.start_mission(first_object['robot_pos'], first_object['color'])
            print(f"🔄 BATCH MODE: {len(objects_list)} objects in queue")
    
    def start_color_batch_pick(self, objects_by_color, color):
        """Start batch picking for all objects of specific color"""
        color_objects = [obj for obj in objects_by_color if obj['color'] == color]
        if color_objects:
            self.start_batch_pick(color_objects)
            print(f"🎯 COLOR BATCH: Picking all {color} objects ({len(color_objects)} found)")
    
    def start_all_objects_batch(self, all_objects):
        """Start batch picking for all detected objects"""
        if all_objects:
            color_order = {"Red": 0, "Green": 1, "Blue": 2}
            sorted_objects = sorted(all_objects, key=lambda x: color_order.get(x['color'], 3))
            self.start_batch_pick(sorted_objects)
            print(f"🚀 FULL BATCH: Picking ALL objects ({len(all_objects)} total)")
    
    def get_current_step(self):
        """Get current mission step based on progress"""
        if not self.current_mission or not self.mission_steps:
            return None
        
        progress_per_step = 100 / len(self.mission_steps)
        current_step_index = min(int(self.mission_progress / progress_per_step), len(self.mission_steps) - 1)
        return self.mission_steps[current_step_index]
    
    def get_status(self):
        """Get current robot status"""
        if self.is_moving and self.current_mission:
            current_step = self.get_current_step()
            queue_info = f" (+{len(self.batch_queue)} in queue)" if self.batch_queue else ""
            return f"MISSION: {self.current_mission['object_color']} - {current_step['description']}{queue_info}"
        else:
            return "READY - Select object to pick"
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        total_time = sum([mission.get('duration', 0) for mission in self.pick_history[-10:]])
        avg_time = total_time / max(len(self.pick_history[-10:]), 1)
        
        return {
            **self.metrics,
            "recent_avg_time": avg_time,
            "total_missions": len(self.pick_history),
            "success_rate": (self.metrics["successful_picks"] / max(self.metrics["total_picks"], 1)) * 100
        }