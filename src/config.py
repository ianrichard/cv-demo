class YOLOConfig:
    # YOLO model files
    WEIGHTS_PATH = 'models/yolov3.weights'
    CONFIG_PATH = 'models/yolov3.cfg'
    
    INPUT_SOURCE = 0
    
    # Path to the classes file
    CLASSES_FILE = 'src/data/yolo_classes.txt'
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.6
    NMS_THRESHOLD = 0.3
    INPUT_WIDTH = 416
    INPUT_HEIGHT = 416
    
    # Performance settings
    TARGET_FPS = 30
    DETECTION_INTERVAL = 1.0