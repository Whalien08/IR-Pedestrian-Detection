import os
import shutil

# 1. Source (where the files are locked)
src_img_path = '/kaggle/input/datasets/ssabana/source/dataset/images'
src_lbl_path = '/kaggle/input/datasets/ssabana/source/dataset/labels'

# 2. Destination (where you have permission to work)
dest_base = '/kaggle/working/dataset'
dest_img_path = os.path.join(dest_base, 'images')
dest_lbl_path = os.path.join(dest_base, 'labels')

os.makedirs(dest_img_path, exist_ok=True)
os.makedirs(dest_lbl_path, exist_ok=True)

# 3. Get and sort files
images = sorted([f for f in os.listdir(src_img_path) if f.endswith('.png')])
labels = sorted([f for f in os.listdir(src_lbl_path) if f.endswith('.txt')])

print(f"📦 Copying and renaming {len(images)} pairs to /kaggle/working...")

# 4. Copy and rename at the same time
count = 0
for img_file in images:
    base_name = os.path.splitext(img_file)[0]
    matching_label = base_name + ".txt"
    
    if matching_label in labels:
        new_name = f"pedestrian_{count}"
        
        # Copy Image from Input -> Working with NEW name
        shutil.copy(os.path.join(src_img_path, img_file), 
                    os.path.join(dest_img_path, new_name + ".png"))
        
        # Copy Label from Input -> Working with NEW name
        shutil.copy(os.path.join(src_lbl_path, matching_label), 
                    os.path.join(dest_lbl_path, new_name + ".txt"))
        count += 1

print(f"✅ Successfully created {count} paired files in /kaggle/working/dataset")