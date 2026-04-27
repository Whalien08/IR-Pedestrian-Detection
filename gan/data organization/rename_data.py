import os

rgb_dir = r"./gan/data/rgb"
thermal_dir = r"./gan/data/thermal"

def rename_files(directory):
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for index, filename in enumerate(files):
        extension = os.path.splitext(filename)[1]
        new_name = f"{index + 1}{extension}"
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
    return len(files)

# Safety Check
rgb_count = len(os.listdir(rgb_dir))
thermal_count = len(os.listdir(thermal_dir))

if rgb_count == thermal_count:
    print("Renaming RGB files...")
    rename_files(rgb_dir)

    print("Renaming Thermal files...")
    rename_files(thermal_dir)
    print(f"Success! Paired {rgb_count} images.")
else:
    print(f"ERROR: Folder mismatch! RGB has {rgb_count}, Thermal has {thermal_count}.")