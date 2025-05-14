class YOLOConfig:
    # YOLO model files
    WEIGHTS_PATH = 'models/yolov3.weights'
    CONFIG_PATH = 'models/yolov3.cfg'
    
    # Camera settings
    INPUT_SOURCE = 0
    MAX_CAMERA_WIDTH = 1280  # Maximum width to process (will resize larger inputs)
    MAX_CAMERA_HEIGHT = 720  # Maximum height to process (will resize larger inputs)
    
    # Path to the classes file
    CLASSES_FILE = 'src/data/yolo_classes.txt'
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    INPUT_WIDTH = 416
    INPUT_HEIGHT = 416
    
    # Detection processing resolution (smaller = faster)
    YOLO_PROCESS_WIDTH = 416  # Standard for YOLO
    YOLO_PROCESS_HEIGHT = 416
    FACE_PROCESS_WIDTH = 320  # Smaller for face detection
    FACE_PROCESS_HEIGHT = 240
    
    # Performance settings
    TARGET_FPS = 30
    DETECTION_INTERVAL = 1.0