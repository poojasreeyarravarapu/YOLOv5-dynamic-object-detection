# YOLOv5-dynamic-object-detection

# Dynamic Object Detection using YOLOv5 & OpenCV

**“What is YOLOv5?”**
YOLO stands for You Only Look Once. It’s a real‑time object detection algorithm that detects objects in an image or video in a single forward pass through a neural network. YOLOv5 is the PyTorch‑based implementation by Ultralytics. It’s fast and accurate for multiple object categories.

**“How does it work?”**
It splits the image into a grid, predicts bounding boxes and class probabilities for each grid cell, and then applies Non‑Max Suppression to remove overlapping boxes and keep the most confident ones.

This project implements a YOLOv5-based robust object detection system for videos.

## Features
- 🎯 **Accurate detection** using YOLOv5
- 🧹 **Automated dataset cleanup** (removes blurry, duplicate, and invalid frames)
- 🎥 **Video to frames** conversion with OpenCV
- 📦 **Final processed video** output

## Setup
1. Install python and pyTorch:
https://www.python.org/
https://pytorch.org/
Follow some youtube tutorials if you don't know how to download!


2. Clone YOLOv5 from Ultralytics:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```
Test if it works
```bash
python detect.py --weights yolov5s.pt --source data/images
```
The results will be saved here:
```bash
runs/detect/exp
```
📌 At this point → You have YOLOv5-based robust object detection

-> If you want to check for your webcam and detect your camera objects:
  ```bash
  python detect.py --weights yolov5s.pt --source 0
  ```
-> For any video in yolov5 folder
  ```bash
  python detect.py --weights yolov5s.pt --source video.mp4
  ```
where myvideo is just the name of the video which you want to apply this. Make sure you have that video present in yolov5 folder.
If you want a sample video download it from this repository named sample_video and download it to yolov5 folder rename it to video.

-> Output will be saved to:
```bash
runs/detect/exp
```

If you want YOLOv5 to show the detection video live while running:
```bash
python detect.py --weights yolov5s.pt --source video.mp4 --view-img
```


3. Place these scripts from this repository inside the yolov5 folder.
     cleanup.py
     dynamic_object_detection.py
     video_to_yolo.py


4. Install required libraries:
   ```bash  
   pip install opencv-python pillow imagehash
   ```

5. For the Automated Data Cleanup:
  For Images:
  cleanup.py - Uses image hashing and blurriness detection to remove low‑quality or duplicate frames.
  ```bash
  python cleanup.py
  ```
  then the file name to be entered is the path to the yolov5 folder like this C:\Users\POOJA SREE\yolov5\ and then:
  ```bash
  data\images
  ```
  Basically the file path of the data\images that need to be cleaned.

  For Videos:
  video_to_yolo.py - Uses OpenCV to read video frames and save them as images.
  And the cleanup script work directly on frames extracted from your video
  Basically to extract frames
  ```bash
  python video_to_yolo.py
  ```
  And enter the path for our video.mp4 in yolov5 folder yolov5\video.mp4
  Output stored in runs\detect\video_clean_detect


6. And for combining all steps into one clean script to run it
   dynamic_object_detection.py - Calls YOLOv5 detection with the provided video/image path, processes the results, and saves them.
  ```bash
  python dynamic_object_detection.py
  ```
  You can give any path wither video or image where you need that object detection to take place.
  The output is stored in:
  ```bash
  runs/detect/video_detection_processed.mp4
  ```


📌 Notes
This project requires Python 3.8+.

If you don't have a GPU, install the CPU-only version of PyTorch.

Pretrained YOLOv5 weights (yolov5s.pt) will be downloaded automatically the first time you run detection.

Keep videos small for faster processing.


License:
This project uses YOLOv5 by Ultralytics under the GNU General Public License v3.0.

```markdown
Finally this is how it works!
┌───────────────────┐
│ Input File │
│ (Video / Image) │
└─────────┬─────────┘
│
▼
┌───────────────────┐
│ Frame Extraction │
│ (video_to_yolo.py) │
└─────────┬─────────┘
│
▼
┌───────────────────┐
│ Dataset Cleanup │
│ (cleanup.py) │
└─────────┬─────────┘
│
▼
┌───────────────────┐
│ YOLOv5 Detection │
│ (dynamic_object_ │
│ detection.py) │
└─────────┬─────────┘
│
▼
┌───────────────────┐
│ Output Video/ │
│ Images (runs/) │
└───────────────────┘

```

This project demonstrates a practical and efficient pipeline for dynamic object detection using YOLOv5 and OpenCV.
By combining video frame extraction, automated dataset cleanup, and YOLOv5 detection, it ensures high‑quality inputs and accurate detection results for both images and videos.
It can be extended to real‑time detection, multi‑camera setups, or integrated into larger computer vision systems.


Author

Pooja Sree Yarravarapu



