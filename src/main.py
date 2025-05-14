import cv2
import time
import numpy as np
from detection.model import YOLOModel
from detection.face_model import FaceModel
from visualization.draw import draw_bounding_boxes
from visualization.text_utils import add_status_text
from config import YOLOConfig
from app.app_manager import AppManager
from app.camera_manager import CameraManager
from app.detection_manager import DetectionManager

def create_letterboxed_frame(frame, target_width, target_height, fill_screen=False):
    """Create a letterboxed/pillarboxed frame that maintains aspect ratio
    
    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height
        fill_screen: If True, fills the entire screen by cropping if necessary
    """
    if frame is None:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    if fill_screen:
        # Fill the entire screen even if it means cropping
        # Calculate scaling factor to ensure minimum dimension covers the screen
        scale = max(target_width / w, target_height / h)
        
        # Calculate new dimensions (may be larger than target)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame to fill screen
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate crop region to center the frame
        start_x = (new_w - target_width) // 2 if new_w > target_width else 0
        start_y = (new_h - target_height) // 2 if new_h > target_height else 0
        
        # Crop to target size
        if new_w >= target_width and new_h >= target_height:
            return resized_frame[start_y:start_y+target_height, start_x:start_x+target_width]
        
        # If we still don't cover the full area (shouldn't happen with max scaling)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_w) // 2 if new_w < target_width else 0
        y_offset = (target_height - new_h) // 2 if new_h < target_height else 0
        
        # Only copy what fits
        copy_w = min(new_w, target_width)
        copy_h = min(new_h, target_height)
        
        canvas[y_offset:y_offset+copy_h, x_offset+x_offset+copy_w] = resized_frame[0:copy_h, 0:copy_w]
        return canvas
    
    else:
        # Original letterbox behavior - add black bars but preserve ratio
        # Create a black canvas of target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_width / w, target_height / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame while maintaining aspect ratio
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate position to center the frame (letterbox/pillarbox)
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        
        # Place the resized frame on the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        
        return canvas

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
        
        # Create a resizable window
        window_name = 'Object Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set initial window size (can be adjusted by user)
        initial_width = YOLOConfig.MAX_CAMERA_WIDTH
        initial_height = YOLOConfig.MAX_CAMERA_HEIGHT
        cv2.resizeWindow(window_name, initial_width, initial_height)
        
        # Main processing loop
        last_frame_time = 0
        frame_interval = 1.0 / YOLOConfig.TARGET_FPS
        
        # Store the window dimensions
        window_width = initial_width
        window_height = initial_height
        
        # Set flags
        is_fullscreen = False
        fill_screen = True  # Default to filling the entire screen in fullscreen mode
        
        while app.running:
            # Check window size each frame
            try:
                curr_width = int(cv2.getWindowProperty(window_name, cv2.WND_PROP_WIDTH))
                curr_height = int(cv2.getWindowProperty(window_name, cv2.WND_PROP_HEIGHT))
                if curr_width > 0 and curr_height > 0 and (curr_width != window_width or curr_height != window_height):
                    window_width, window_height = curr_width, curr_height
                    print(f"Window resized to {window_width}x{window_height}")
            except:
                pass
            
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
                
                # Create display frame with appropriate mode
                # In fullscreen, use fill_screen mode to cover the entire area
                display_frame = create_letterboxed_frame(
                    output_frame, 
                    window_width, 
                    window_height,
                    fill_screen=is_fullscreen and fill_screen
                )
                
                # Display the frame
                cv2.imshow(window_name, display_frame)

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
                elif key == ord('m'):  # Toggle fullscreen mode
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print(f"Fullscreen mode: {'On' if is_fullscreen else 'Off'}")
                elif key == ord('a'):  # Toggle aspect ratio mode (fill vs letterbox)
                    fill_screen = not fill_screen
                    print(f"Fill screen mode: {'On' if fill_screen else 'Off'}")

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