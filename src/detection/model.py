import cv2
import numpy as np
from config import YOLOConfig

class YOLOModel:
    def __init__(self):
        self.net = None
        self.output_layers = None
        
    def load_model(self, weights_path, config_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            (YOLOConfig.INPUT_WIDTH, YOLOConfig.INPUT_HEIGHT), 
            swapRB=True, 
            crop=False
        )
        self.net.setInput(blob)
        
        # Forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > YOLOConfig.CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                  YOLOConfig.CONFIDENCE_THRESHOLD, 
                                  YOLOConfig.NMS_THRESHOLD)
        
        return boxes, class_ids, confidences