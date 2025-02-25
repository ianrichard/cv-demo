import queue

class FaceDetectorThread:
    def __init__(self, face_model, input_queue, result_queue, running_flag):
        self.face_model = face_model
        self.input_queue = input_queue
        self.result_queue = result_queue
        self.running_flag = running_flag
    
    def run(self):
        """Worker function that runs in a thread for face detection"""
        print("Face detection thread started")
        while self.running_flag:
            try:
                # Try to get a frame with a short timeout
                try:
                    frame = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the frame for face detection
                try:
                    face_results = self.face_model.detect_objects(frame)
                    
                    # Put results in the output queue, but don't block
                    if not self.result_queue.full():
                        self.result_queue.put(face_results)
                        
                    face_boxes = face_results[0]
                    if face_boxes:
                        print(f"Thread detected {len(face_boxes)} faces")
                except Exception as e:
                    print(f"Face thread error: {str(e)}")
            except Exception as e:
                print(f"Face thread general error: {str(e)}")
                
        print("Face detection thread ended")