import cv2
import numpy as np
from config import YOLOConfig

class YOLOModel:
    def __init__(self):
        self.net = None
        self.output_layers = None
        
    def load_model(self, weights_path, config_path):
        try:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Mac-specific optimizations
            # Try OpenCL first, fallback to CPU if not available
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                print("Using OpenCL backend")
            except:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU backend")
                
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        
        # Load enabled classes
        with open(YOLOConfig.CLASSES_FILE, 'r') as f:
            enabled_classes = [i for i, line in enumerate(f.readlines()) 
                             if not line.strip().startswith('#')]
        
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            (YOLOConfig.INPUT_WIDTH, YOLOConfig.INPUT_HEIGHT), 
            swapRB=True, 
            crop=False
        )
        self.net.setInput(blob)
        
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only process if class is enabled and confidence is high enough
                if (class_id in enabled_classes and 
                    confidence > YOLOConfig.CONFIDENCE_THRESHOLD):
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply NMS
        if boxes:  # Only if we have detections
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                confidences, 
                YOLOConfig.CONFIDENCE_THRESHOLD, 
                YOLOConfig.NMS_THRESHOLD
            )
            
            # Filter results based on NMS
            filtered_boxes = [boxes[i] for i in indices]
            filtered_class_ids = [class_ids[i] for i in indices]
            filtered_confidences = [confidences[i] for i in indices]
        else:
            filtered_boxes = []
            filtered_class_ids = []
            filtered_confidences = []
        
        return filtered_boxes, filtered_class_ids, filtered_confidences