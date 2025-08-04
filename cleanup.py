import os
import cv2
from PIL import Image
import imagehash

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

        # 1️⃣ Remove unreadable or invalid images
        if not is_image(path):
            os.remove(path)
            removed_files += 1
            print(f"❌ Removed invalid or unreadable: {file}")
            continue

        # 2️⃣ Remove blurry images
        img = cv2.imread(path)
        if cv2.Laplacian(img, cv2.CV_64F).var() < blur_threshold:
            os.remove(path)
            removed_files += 1
            print(f"❌ Removed blurry image: {file}")
            continue

        # 3️⃣ Remove duplicate images
        hash_val = imagehash.average_hash(Image.open(path))
        if hash_val in seen_hashes:
            os.remove(path)
            removed_files += 1
            print(f"❌ Removed duplicate: {file}")
        else:
            seen_hashes.add(hash_val)

    print("\n✅ Cleanup Completed!")
    print(f"Total files scanned: {total_files}")
    print(f"Total files removed: {removed_files}")
    print(f"Total files kept: {total_files - removed_files}")

if __name__ == "__main__":
    folder = input("Enter path to dataset folder: ").strip('"')
    clean_dataset(folder)
