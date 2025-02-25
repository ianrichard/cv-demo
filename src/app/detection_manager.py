import os
import time
import queue
from utils.result_processor import ResultProcessor

class DetectionManager:
    def __init__(self, app_manager, yolo_model, face_model):
        self.app_manager = app_manager
        self.yolo_model = yolo_model
        self.face_model = face_model
        
        self.last_yolo_detection_time = 0
        self.last_face_submission_time = 0
        self.face_interval = 2.0  # Submit frames for face detection every 2 seconds
        
        # Store last detection results
        self.last_yolo_results = ([], [], [])
        self.last_face_results = ([], [], [])
        
        # Load classes
        self.yolo_classes = []
        self.face_classes = []
        self.classes = []
        self.load_classes()
        
        self.result_processor = ResultProcessor()
    
    def load_classes(self, faces_dir="training_data/faces"):
        """Load YOLO and face recognition classes"""
        from config import YOLOConfig
        
        # Load YOLO classes
        with open(YOLOConfig.CLASSES_FILE, 'r') as f:
            self.yolo_classes = f.read().strip().split('\n')
        
        # Load face classes
        self.face_classes = []
        if os.path.exists(faces_dir):
            self.face_classes = [os.path.splitext(f)[0] for f in os.listdir(faces_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Combine classes
        self.classes = self.yolo_classes + self.face_classes
        
        print(f"YOLO classes: {len(self.yolo_classes)}")
        print(f"Face classes: {self.face_classes}")
        
        return self.classes
    
    def process_frame(self, frame, current_time):
        """Process a frame with YOLO and face detection"""
        from config import YOLOConfig
        
        # YOLO detection at normal interval (if enabled)
        if self.app_manager.object_detection_enabled and current_time - self.last_yolo_detection_time >= YOLOConfig.DETECTION_INTERVAL:
            self.last_yolo_results = self.yolo_model.detect_objects(frame)
            self.last_yolo_detection_time = current_time
        elif not self.app_manager.object_detection_enabled:
            # Clear YOLO results when disabled
            self.last_yolo_results = ([], [], [])
        
        # Submit frames for face detection at specified interval (if enabled)
        if self.app_manager.face_detection_enabled and current_time - self.last_face_submission_time >= self.face_interval:
            if self.app_manager.face_queue.empty():  # Only if queue is empty
                self.app_manager.face_queue.put(frame.copy())
                self.last_face_submission_time = current_time
        elif not self.app_manager.face_detection_enabled:
            # Clear face results when disabled
            self.last_face_results = ([], [], [])
        
        # Check for face detection results
        try:
            if not self.app_manager.face_result_queue.empty():
                self.last_face_results = self.app_manager.face_result_queue.get_nowait()
        except queue.Empty:
            pass  # No results available yet
        
        # Return combined detection results
        return self.result_processor.combine_results(
            self.last_yolo_results,
            self.last_face_results,
            self.yolo_classes,
            self.classes
        )