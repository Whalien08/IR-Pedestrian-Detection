import os
import shutil

# --- CONFIGURATION ---
source_folder = r"./raw_data/JPEGImages" 
rgb_out = r"./gan/data/rgb"
thermal_out = r"./gan/data/thermal"

os.makedirs(rgb_out, exist_ok=True)
os.makedirs(thermal_out, exist_ok=True)

all_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"Found {len(all_files)} files. Starting sort...")

rgb_count = 0
ir_count = 0

# Using enumerate(all_files) is much faster than all_files.index(filename)
for index, filename in enumerate(all_files):
    fn_lower = filename.lower()
    source_path = os.path.join(source_folder, filename)
    
    # Priority 1: Name-based sorting (Safest)
    if any(x in fn_lower for x in ['rgb', 'visible', 'v_']):
        shutil.copy(source_path, os.path.join(rgb_out, filename))
        rgb_count += 1
    elif any(x in fn_lower for x in ['ir', 'infrared', 'thermal', 'i_']):
        shutil.copy(source_path, os.path.join(thermal_out, filename))
        ir_count += 1
    
    # Priority 2: Alternating logic (Use only if names are just numbers)
    else:
        if index % 2 == 0:
            shutil.copy(source_path, os.path.join(rgb_out, filename))
            rgb_count += 1
        else:
            shutil.copy(source_path, os.path.join(thermal_out, filename))
            ir_count += 1

print(f"Done! RGB: {rgb_count} | Thermal: {ir_count}")
if rgb_count != ir_count:
    print(" WARNING: Counts do not match! Check for missing image pairs.")