# Computer Vision Demo

A simple real-time object detection demo using YOLO and OpenCV. Uses your webcam to detect objects and draw labeled boxes around them.

## Quick Start

1. Clone and setup:
```bash
git clone git@github.com:ianrichard/cv-demo.git
cd cv-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download YOLO model files:
```bash
mkdir models
cd models
curl -O https://pjreddie.com/media/files/yolov3.weights
curl -O https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
cd ..
```

3. Run the demo:
```bash
python src/main.py
```

## Controls
- Press 'c' to switch between cameras (built-in/USB)
- Press 'q' to quit

## Testing Camera Sources
To find available cameras:
```bash
python src/utils/camera_test.py
```
Then update `INPUT_SOURCE` in `src/config.py` with your preferred camera index (usually 0 or 1).