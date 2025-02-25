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