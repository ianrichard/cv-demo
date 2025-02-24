import cv2

def draw_bounding_boxes(image, boxes, class_ids, confidences, classes):
    """
    Draws bounding boxes and labels on the input image.

    Parameters:
    - image: The input image on which to draw.
    - boxes: A list of bounding box coordinates (x, y, width, height).
    - class_ids: A list of class IDs corresponding to the detected objects.
    - confidences: A list of confidence scores for the detections.
    - classes: A list of class names corresponding to the class IDs.
    """
    for i in range(len(boxes)):
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        
        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Put the label above the bounding box
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image