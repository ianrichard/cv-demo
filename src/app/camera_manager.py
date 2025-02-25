import cv2
from config import YOLOConfig

class CameraManager:
    def __init__(self, app_manager):
        self.app_manager = app_manager
        self.current_camera = YOLOConfig.INPUT_SOURCE
        
    def setup_camera(self, camera_id):
        """Initialize camera with error handling"""
        if self.app_manager.cap:
            self.app_manager.cap.release()
        
        self.app_manager.cap = cv2.VideoCapture(camera_id)
        if not self.app_manager.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        self.app_manager.cap.set(cv2.CAP_PROP_FPS, YOLOConfig.TARGET_FPS)
        return self.app_manager.cap
    
    def switch_camera(self):
        """Switch between available cameras"""
        self.current_camera = 1 if self.current_camera == 0 else 0
        return self.setup_camera(self.current_camera)
        
    def get_frame(self):
        """Get a frame from the camera"""
        if not self.app_manager.cap:
            return False, None
        return self.app_manager.cap.read()