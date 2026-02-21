import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Load the model
model_path = '/kaggle/input/notebooks/himawariricttoslock/gan-4-3/saved_models/thermal_gen_best.h5'
model = tf.keras.models.load_model(model_path, compile=False)

def display_thermal_results(input_rgb_folder, num_to_show=8):
    img_list = [f for f in os.listdir(input_rgb_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_list = img_list[:num_to_show] # Limit the number of images displayed
    
    # Calculate grid size 
    cols = 2 
    rows = (len(img_list) + 1) // 2
    plt.figure(figsize=(15, rows * 5))

    for i, img_name in enumerate(tqdm(img_list)):
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

        # Plotting
        plt.subplot(rows, cols, i + 1)
        plt.imshow(thermal_data, cmap='grey') # 'inferno' or 'magma' gives a realistic thermal look
        plt.title(f"Thermal: {img_name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run it to show the first 6 images
display_thermal_results('/kaggle/input/inputttttttttttttttttttttt/', num_to_show=9)
 