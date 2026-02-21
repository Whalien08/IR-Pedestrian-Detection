import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 1. Load the model
model_path = r'C:\Users\Nyx\Desktop\IR Pedestrian Detection\gan\final weights\thermal_gan_best.h5'
model = tf.keras.models.load_model(model_path, compile=False)

def save_thermal_results(input_rgb_folder, output_ir_folder):
    # Ensure output directory exists
    if not os.path.exists(output_ir_folder):
        os.makedirs(output_ir_folder)

    img_list = [f for f in os.listdir(input_rgb_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(img_list)} images...")

    for img_name in tqdm(img_list):
        path = os.path.join(input_rgb_folder, img_name)
        img = cv2.imread(path)
        if img is None: continue
        
        # Preprocessing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        input_arr = (img_resized / 127.5) - 1.0
        
        # Inference
        prediction = model.predict(np.expand_dims(input_arr, 0), verbose=0)
        
        # Post-processing
        thermal_data = np.mean(prediction[0], axis=-1)
        thermal_data = ((thermal_data + 1.0) * 127.5).astype(np.uint8)
        thermal_data = cv2.medianBlur(thermal_data, 5)
        
        # Intensity reduction for cooler areas
        threshold = 100
        thermal_data[thermal_data < threshold] = np.clip(thermal_data[thermal_data < threshold] * 0.2, 0, 255)

        # --- SAVE IMAGE LOGIC ---
        save_path = os.path.join(output_ir_folder, f"ir_{img_name}")
        cv2.imwrite(save_path, thermal_data)

    print(f"\nDone! All generated IR images saved to: {output_ir_folder}")

# Set your paths
input_dir = r'C:\Users\Nyx\Desktop\IR Pedestrian Detection\gan\data\generate\rgb images'
output_dir = r'C:\Users\Nyx\Desktop\IR Pedestrian Detection\gan\data\generate\output ir'

# Run it
save_thermal_results(input_dir, output_dir)