import cv2
import time
import numpy as np
from detection.model import YOLOModel
from detection.face_model import FaceModel
from visualization.draw import draw_bounding_boxes
from visualization.text_utils import add_status_text  # Import the new utility
from config import YOLOConfig
from app.app_manager import AppManager
from app.camera_manager import CameraManager
from app.detection_manager import DetectionManager

def main():
    app = AppManager()
    
    try:
        # Initialize models
        yolo_model = YOLOModel()
        yolo_model.load_model(YOLOConfig.WEIGHTS_PATH, YOLOConfig.CONFIG_PATH)
        
        face_model = FaceModel()
        face_model.load_faces("training_data/faces")

        if face_model.known_face_encodings:
            print(f"Successfully loaded {len(face_model.known_face_encodings)} face encodings")
        else:
            print("WARNING: No face encodings loaded")

        # Start the face detection thread
        app.start_face_thread(face_model)

        # Initialize manager components
        camera_mgr = CameraManager(app)
        detection_mgr = DetectionManager(app, yolo_model, face_model)
        
        # Setup camera
        cap = camera_mgr.setup_camera(camera_mgr.current_camera)
        
        # Main processing loop
        last_frame_time = 0
        frame_interval = 1.0 / YOLOConfig.TARGET_FPS
        
        while app.running:
            ret, frame = camera_mgr.get_frame()
            if not ret:
                print(f"Failed to read from camera {camera_mgr.current_camera}")
                break

            current_time = time.time()

            # Control frame display rate
            if current_time - last_frame_time < frame_interval:
                continue
            last_frame_time = current_time

            try:
                # Process frame for detections
                boxes, class_ids, confidences = detection_mgr.process_frame(frame, current_time)

                # Draw results
                output_frame = draw_bounding_boxes(
                    frame.copy(),
                    boxes, 
                    class_ids, 
                    confidences, 
                    detection_mgr.classes
                )

                # Add status text - using the imported utility
                output_frame = add_status_text(output_frame, app)

                cv2.imshow('Object Detection', output_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('Quit command received')
                    break
                elif key == ord('c'):
                    cap = camera_mgr.switch_camera()
                elif key == ord('f'):
                    app.toggle_face_detection()
                elif key == ord('o'):
                    app.toggle_object_detection()

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