import os
import shutil
from ultralytics import YOLO

# ===== 1. PATHS =====
# Update these to match your exact Kaggle Input sidebar
INPUT_BASE = "/kaggle/input/datasets/ssabana/source/dataset"
IR_SOURCE = os.path.join(INPUT_BASE, "images")
LABELS_SOURCE = os.path.join(INPUT_BASE, "labels")

# Path to your previous best.pt weights from static training
PREVIOUS_MODEL = "/kaggle/input/datasets/ssabana/bestbest/best.pt"

# ===== 2. PREPARE WORKING DIRECTORY =====
TRAIN_DIR = "/kaggle/working/yolo_data"
if os.path.exists(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)

os.makedirs(f"{TRAIN_DIR}/images/train", exist_ok=True)
os.makedirs(f"{TRAIN_DIR}/labels/train", exist_ok=True)

# ===== 3. DIRECT PAIRING LOGIC =====
print("🔄 Pairing matching filenames...")
count = 0

# We iterate through labels and look for the exact same name in images
for lbl_file in os.listdir(LABELS_SOURCE):
    if lbl_file.endswith(".txt"):
        # Image name is identical, just with .png
        img_file = lbl_file.replace(".txt", ".png")
        src_img = os.path.join(IR_SOURCE, img_file)
        
        if os.path.exists(src_img):
            # Copy both to the training directory
            shutil.copy(os.path.join(LABELS_SOURCE, lbl_file), f"{TRAIN_DIR}/labels/train/{lbl_file}")
            shutil.copy(src_img, f"{TRAIN_DIR}/images/train/{img_file}")
            count += 1

print(f"✅ Successfully paired {count} files. Folder structure is ready.")

# ===== 4. TRAIN WITH PREVIOUS BEST.PT =====
if count > 0:
    # Create the config file YOLO needs
    with open('/kaggle/working/data.yaml', 'w') as f:
        f.write(f"path: {TRAIN_DIR}\ntrain: images/train\nval: images/train\nnc: 1\nnames: ['person']")

    # Load your previous 93.7% mAP model
    print(f"🚀 Loading weights from previous training...")
    model = YOLO(PREVIOUS_MODEL)
    
    # Fine-tune on the new video frames
    model.train(
        data="/kaggle/working/data.yaml",
        epochs=50,
        imgsz=512,
        batch=16,
        device=0,
        name="thermal_video_finetuning"
    )