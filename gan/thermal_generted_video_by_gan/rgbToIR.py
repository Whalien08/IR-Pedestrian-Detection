import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# --- Your existing GAN functions (Keep these exactly as they are) ---
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False, safe_mode=True)

def preprocess_image(img, target_size=(512, 512)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    return (img_resized / 127.5) - 1.0

def postprocess_thermal_data(prediction):
    thermal_data = np.mean(prediction[0], axis=-1)
    thermal_data = ((thermal_data + 1.0) * 127.5).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    thermal_data = clahe.apply(thermal_data)
    thermal_data = cv2.medianBlur(thermal_data, 3)
    gaussian_3 = cv2.GaussianBlur(thermal_data, (0, 0), 2.0)
    thermal_data = cv2.addWeighted(thermal_data, 1.5, gaussian_3, -0.5, 0)
    return thermal_data

# --- NEW: Video Processing Function ---
def convert_video_to_ir(input_video_path, model, output_path='/kaggle/working/output_thermal.mp4'):
    # 1. Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 2. Get video metadata
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = 512  # We match your GAN target size
    height = 512
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Define the Codec and create VideoWriter
    # 'mp4v' is standard for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    print(f"Processing {total_frames} frames...")

    # 4. Loop through frames
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        input_arr = preprocess_image(frame)
        
        # GAN Inference
        prediction = model.predict(np.expand_dims(input_arr, 0), verbose=0)
        
        # Post-process to Thermal Grayscale
        thermal_frame = postprocess_thermal_data(prediction)
        
        # Write the frame to the new video
        out.write(thermal_frame)

    # 5. Release everything
    cap.release()
    out.release()
    print(f"Video saved successfully to: {output_path}")

# --- RUN THE VIDEO CONVERSION ---
model_path = '/kaggle/input/datasets/ssabana/22-epoch/thermal_gen_epoch_22.h5'
input_video = '/kaggle/input/datasets/ssabana/video-2/mixkit-subway-network-in-tokyo-4453-hd-ready.mp4' # Replace with your video path

model = load_model(model_path)
if model:
    convert_video_to_ir(input_video, model)