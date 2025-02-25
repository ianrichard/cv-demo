class ResultProcessor:
    def combine_results(self, yolo_results, face_results, yolo_classes, classes=None):
        """Combine YOLO and face detection results"""
        boxes = list(yolo_results[0])
        class_ids = list(yolo_results[1]) 
        confidences = list(yolo_results[2])
        
        # Adjust face class IDs and add face detections
        face_boxes, face_class_ids, face_confidences = face_results
        for i, face_id in enumerate(face_class_ids):
            if face_id >= 0:  # Valid detection
                boxes.append(face_boxes[i])
                class_ids.append(len(yolo_classes) + face_id)  # Offset by YOLO classes
                confidences.append(face_confidences[i])
                
        return boxes, class_ids, confidences