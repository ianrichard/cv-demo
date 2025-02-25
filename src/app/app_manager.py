import cv2
import signal
import threading
import queue
from detection.detector_thread import FaceDetectorThread

class AppManager:
    def __init__(self):
        self.running = True
        self.cap = None
        self.setup_signal_handlers()
        
        # Feature toggle flags
        self.face_detection_enabled = True
        self.object_detection_enabled = True
        
        # Threading components
        self.face_queue = queue.Queue(maxsize=1)
        self.face_result_queue = queue.Queue(maxsize=1)
        self.face_thread = None
        self.face_thread_running = False

    def setup_signal_handlers(self):
        """Set up handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print('\nShutting down gracefully...')
        self.running = False

    def cleanup(self):
        """Clean up resources"""
        # Stop face detection thread
        self.face_thread_running = False
        if self.face_thread and self.face_thread.is_alive():
            self.face_thread.join(timeout=1.0)
            
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print('Cleanup complete')
        
    def start_face_thread(self, face_model):
        """Start the face detection thread"""
        self.face_thread_running = True
        detector = FaceDetectorThread(
            face_model,
            self.face_queue, 
            self.face_result_queue, 
            self.face_thread_running
        )
        self.face_thread = threading.Thread(
            target=detector.run,
            daemon=True
        )
        self.face_thread.start()
        
    def toggle_face_detection(self):
        """Toggle face detection on/off"""
        self.face_detection_enabled = not self.face_detection_enabled
        status = "enabled" if self.face_detection_enabled else "disabled"
        print(f"Face detection {status}")
        return self.face_detection_enabled
        
    def toggle_object_detection(self):
        """Toggle object detection on/off"""
        self.object_detection_enabled = not self.object_detection_enabled
        status = "enabled" if self.object_detection_enabled else "disabled"
        print(f"Object detection {status}")
        return self.object_detection_enabled