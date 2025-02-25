import face_recognition
import cv2
import os
import glob
import numpy as np
from PIL import Image

class FaceModel:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_faces(self, faces_dir="training_data/faces"):
        """Load all image files directly from faces directory"""
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print(f"Created faces directory at {faces_dir}")
            return

        image_files = glob.glob(os.path.join(faces_dir, "*.[jp][pn][g]"))
        
        for image_path in image_files:
            try:
                # Use filename without extension as person's name
                name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Load image using PIL first to ensure correct format
                image = Image.open(image_path)
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                # Convert to numpy array
                image_np = np.array(image)
                
                # Detect faces in the image
                face_locations = face_recognition.face_locations(image_np)
                if not face_locations:
                    print(f"No face found in {image_path}")
                    continue
                
                # Get face encoding for the first face found
                face_encoding = face_recognition.face_encodings(
                    image_np,
                    known_face_locations=[face_locations[0]]
                )[0]
                
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                print(f"Loaded face: {name}")
                
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")
    
    def detect_objects(self, frame):
        """Detect and identify faces in frame"""
        if frame is None or frame.size == 0:
            return [], [], []
            
        try:
            # Ensure frame is in RGB
            if len(frame.shape) == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return [], [], []
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_frame)
            if not face_locations:
                return [], [], []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations
            )
            
            boxes = []
            class_ids = []
            confidences = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Create bounding box
                boxes.append([
                    left,
                    top,
                    right - left,
                    bottom - top
                ])
                
                if len(self.known_face_encodings) > 0:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=0.6
                    )
                    
                    if True in matches:
                        class_ids.append(matches.index(True))
                        confidences.append(0.99)
                    else:
                        class_ids.append(-1)
                        confidences.append(0.5)
                else:
                    class_ids.append(-1)
                    confidences.append(0.5)
            
            return boxes, class_ids, confidences
            
        except Exception as e:
            print(f"Error in detect_objects: {str(e)}")
            return [], [], []