import os
import cv2
from PIL import Image
import imagehash
import subprocess
import glob

# ---------- IMAGE CLEANUP ----------
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

        # Remove invalid/unreadable images
        if not is_image(path):
            os.remove(path)
            removed_files += 1
            print(f"❌ Invalid or unreadable: {file}")
            continue

        # Remove blurry images
        img = cv2.imread(path)
        if cv2.Laplacian(img, cv2.CV_64F).var() < blur_threshold:
            os.remove(path)
            removed_files += 1
            print(f"❌ Blurry: {file}")
            continue

        # Remove duplicates
        hash_val = imagehash.average_hash(Image.open(path))
        if hash_val in seen_hashes:
            os.remove(path)
            removed_files += 1
            print(f"❌ Duplicate: {file}")
        else:
            seen_hashes.add(hash_val)

    print(f"\n✅ Cleanup Completed! Removed {removed_files} / {total_files} images.")

# ---------- FRAME EXTRACTION ----------
def extract_frames(video_path, output_folder, frame_skip=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
    print(f"🎥 Extracted {saved} frames from video.")

# ---------- CONVERT DETECTIONS TO VIDEO ----------
def images_to_video(image_folder, output_video_path, fps=20):
    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if not images:
        print("❌ No images found to make a video!")
        return

    first_frame = cv2.imread(images[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"🎬 Video saved: {output_video_path}")

# ---------- MAIN ----------
if __name__ == "__main__":
    video_file = input("Enter full path to video: ").strip('"')
    frames_folder = "frames_dataset"

    # 1. Extract frames
    extract_frames(video_file, frames_folder, frame_skip=5)

    # 2. Cleanup dataset
    clean_dataset(frames_folder, blur_threshold=50)

    # 3. Run YOLOv5 detection
    print("\n🚀 Running YOLOv5 detection on cleaned frames...")
    subprocess.run([
        "python", "detect.py",
        "--weights", "yolov5s.pt",
        "--source", frames_folder,
        "--project", "runs/detect",
        "--name", "video_detection",
        "--exist-ok"
    ])

    # 4. Convert detection results to a single video
    detection_folder = "runs/detect/video_detection"
    output_video = "runs/detect/video_detection_processed.mp4"
    images_to_video(detection_folder, output_video, fps=20)

    print("\n✅ All done! Final processed video ready for presentation.")
