import os
import cv2
from PIL import Image
import imagehash
import subprocess

# ---------- CLEANUP FUNCTIONS ----------
def is_image(path):
    try:
        Image.open(path).verify()
        return True
    except:
        return False

def clean_dataset(folder_path, blur_threshold=50):
    total_files = 0
    removed_files = 0
    seen_hashes = set()

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if not os.path.isfile(path):
            continue

        total_files += 1

        # Remove invalid images
        if not is_image(path):
            os.remove(path)
            removed_files += 1
            print(f"❌ Invalid image: {file}")
            continue

        # Remove blurry images
        img = cv2.imread(path)
        if cv2.Laplacian(img, cv2.CV_64F).var() < blur_threshold:
            os.remove(path)
            removed_files += 1
            print(f"❌ Blurry image: {file}")
            continue

        # Remove duplicates
        hash_val = imagehash.average_hash(Image.open(path))
        if hash_val in seen_hashes:
            os.remove(path)
            removed_files += 1
            print(f"❌ Duplicate: {file}")
        else:
            seen_hashes.add(hash_val)

    print(f"\n✅ Cleanup done! Removed {removed_files} / {total_files} files.")

# ---------- FRAME EXTRACTION ----------
def extract_frames(video_path, output_folder, frame_skip=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
    print(f"🎥 Extracted {saved} frames from video.")

# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    video_file = input("Enter video path: ").strip('"')
    frames_folder = "frames_dataset"

    # 1. Extract frames
    extract_frames(video_file, frames_folder, frame_skip=5)

    # 2. Cleanup frames
    clean_dataset(frames_folder, blur_threshold=50)

    # 3. Run YOLOv5 detection on cleaned frames
    print("\n🚀 Running YOLOv5 detection on cleaned frames...")
    subprocess.run([
        "python", "detect.py",
        "--weights", "yolov5s.pt",
        "--source", frames_folder,
        "--project", "runs/detect",
        "--name", "video_clean_detect",
        "--exist-ok"
    ])
