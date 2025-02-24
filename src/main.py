import cv2
from detection.model import YOLOModel
from visualization.draw import draw_bounding_boxes
from config import YOLOConfig

def main():
    # Initialize YOLO model
    yolo_model = YOLOModel()
    yolo_model.load_model(YOLOConfig.WEIGHTS_PATH, YOLOConfig.CONFIG_PATH)

    # Load class names
    with open(YOLOConfig.CLASSES_FILE, 'r') as f:
        classes = f.read().strip().split('\n')

    # Initialize video capture with current camera
    current_camera = YOLOConfig.INPUT_SOURCE
    cap = cv2.VideoCapture(current_camera)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        boxes, class_ids, confidences = yolo_model.detect_objects(frame)

        # Draw results
        output_frame = draw_bounding_boxes(
            frame, 
            boxes, 
            class_ids, 
            confidences, 
            classes
        )

        # Add camera info to frame
        cv2.putText(output_frame, f"Camera: {current_camera}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_frame, "Press 'c' to switch camera", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('YOLO Object Detection', output_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Switch between cameras 0 and 1
            current_camera = 1 if current_camera == 0 else 0
            cap.release()
            cap = cv2.VideoCapture(current_camera)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()