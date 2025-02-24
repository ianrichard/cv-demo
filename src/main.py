import cv2
import time
import signal
import sys
from detection.model import YOLOModel
from visualization.draw import draw_bounding_boxes
from config import YOLOConfig

class AppManager:
    def __init__(self):
        self.running = True
        self.cap = None
        self.setup_signal_handlers()

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
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print('Cleanup complete')

    def setup_camera(self, camera_id):
        """Initialize camera with error handling"""
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        self.cap.set(cv2.CAP_PROP_FPS, YOLOConfig.TARGET_FPS)
        return self.cap

def main():
    app = AppManager()
    
    try:
        # Initialize YOLO model
        yolo_model = YOLOModel()
        yolo_model.load_model(YOLOConfig.WEIGHTS_PATH, YOLOConfig.CONFIG_PATH)

        # Load class names
        with open(YOLOConfig.CLASSES_FILE, 'r') as f:
            classes = f.read().strip().split('\n')

        current_camera = YOLOConfig.INPUT_SOURCE
        cap = app.setup_camera(current_camera)

        last_detection_time = 0
        last_frame_time = 0
        frame_interval = 1.0 / YOLOConfig.TARGET_FPS
        
        # Store last detection results
        last_boxes, last_class_ids, last_confidences = [], [], []

        while app.running:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read from camera {current_camera}")
                break

            current_time = time.time()

            # Control frame display rate
            if current_time - last_frame_time < frame_interval:
                continue
            last_frame_time = current_time

            try:
                # Only run detection at specified interval
                if current_time - last_detection_time >= YOLOConfig.DETECTION_INTERVAL:
                    # Detect objects
                    last_boxes, last_class_ids, last_confidences = yolo_model.detect_objects(frame)
                    last_detection_time = current_time

                # Draw results using the last detection results
                output_frame = draw_bounding_boxes(
                    frame, 
                    last_boxes, 
                    last_class_ids, 
                    last_confidences, 
                    classes
                )

                cv2.imshow('YOLO Object Detection', output_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('Quit command received')
                    break
                elif key == ord('c'):
                    current_camera = 1 if current_camera == 0 else 0
                    cap = app.setup_camera(current_camera)

            except KeyboardInterrupt:
                print('\nInterrupt received')
                break
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                break

    except Exception as e:
        print(f"Initialization error: {str(e)}")
    finally:
        app.cleanup()
        print('Application terminated')

if __name__ == "__main__":
    main()