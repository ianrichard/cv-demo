import cv2

def list_cameras():
    # Try indices 0 through 10
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera index {i} is available")
            ret, frame = cap.read()
            if ret:
                print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        else:
            print(f"Camera index {i} is not available")

if __name__ == "__main__":
    list_cameras()