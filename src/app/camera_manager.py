import cv2
from config import YOLOConfig

class CameraManager:
    def __init__(self, app_manager):
        self.app_manager = app_manager
        self.current_camera = YOLOConfig.INPUT_SOURCE
        self.original_width = 0
        self.original_height = 0
        self.needs_resize = False
        
    def setup_camera(self, camera_id):
        """Initialize camera with error handling and resolution management"""
        if self.app_manager.cap:
            self.app_manager.cap.release()
        
        self.app_manager.cap = cv2.VideoCapture(camera_id)
        if not self.app_manager.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
            
        # Set camera FPS
        self.app_manager.cap.set(cv2.CAP_PROP_FPS, YOLOConfig.TARGET_FPS)
        
        # Get original camera resolution
        self.original_width = int(self.app_manager.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.app_manager.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera resolution: {self.original_width}x{self.original_height}")
        
        # Check if resizing is needed
        self.needs_resize = (self.original_width > YOLOConfig.MAX_CAMERA_WIDTH or 
                            self.original_height > YOLOConfig.MAX_CAMERA_HEIGHT)
        
        if self.needs_resize:
            print(f"Input will be resized to max {YOLOConfig.MAX_CAMERA_WIDTH}x{YOLOConfig.MAX_CAMERA_HEIGHT}")
            
        return self.app_manager.cap
    
    def switch_camera(self):
        """Switch between available cameras"""
        self.current_camera = 1 if self.current_camera == 0 else 0
        return self.setup_camera(self.current_camera)
        
    def get_frame(self):
        """Get a frame from the camera with optional resizing"""
        if not self.app_manager.cap:
            return False, None
            
        ret, frame = self.app_manager.cap.read()
        if not ret:
            return False, None
            
        # Resize frame if needed (preserve aspect ratio)
        if self.needs_resize:
            # Calculate new dimensions while preserving aspect ratio
            scale = min(YOLOConfig.MAX_CAMERA_WIDTH / self.original_width,
                        YOLOConfig.MAX_CAMERA_HEIGHT / self.original_height)
            new_width = int(self.original_width * scale)
            new_height = int(self.original_height * scale)
            
            # Resize frame
            frame = cv2.resize(frame, (new_width, new_height), 
                              interpolation=cv2.INTER_AREA)  # INTER_AREA is best for downsampling
            
        return True, frame