class YOLOConfig:
    # YOLO model files
    WEIGHTS_PATH = 'models/yolov3.weights'
    CONFIG_PATH = 'models/yolov3.cfg'
    
    # Input source (1 for USB webcam, 0 for built-in camera)
    INPUT_SOURCE = 1
    
    # Path to the classes file
    CLASSES_FILE = 'src/data/yolo_classes.txt'
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    INPUT_WIDTH = 416
    INPUT_HEIGHT = 416