import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tqdm import tqdm

# 1. LOAD BOTH MODELS
# Load your GAN (.h5) and YOLO (.pt)
gan_model = tf.keras.models.load_model('thermal_gen_epoch_100.h5', compile=False)
yolo_model = YOLO('best.pt')

def process_pipeline(input_video_path, output_path='final_detection.mp4'):
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define output video writer (Color=True to see blue YOLO boxes)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512), isColor=True)

    print("🚀 Running Integrated Pipeline...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        # --- STEP A: GAN CONVERSION ---
        # 1. Resize and normalize for GAN
        img_resized = cv2.resize(frame, (512, 512))
        img_input = (img_resized / 127.5) - 1.0
        
        # 2. GAN Generate (Thermal)
        prediction = gan_model.predict(np.expand_dims(img_input, 0), verbose=0)
        
        # 3. Post-process to Grayscale then back to 3-channel for YOLO
        thermal_gray = np.mean(prediction[0], axis=-1)
        thermal_gray = ((thermal_gray + 1.0) * 127.5).astype(np.uint8)
        
        # YOLOv8 prefers 3-channel BGR images
        thermal_bgr = cv2.cvtColor(thermal_gray, cv2.COLOR_GRAY2BGR)

        # --- STEP B: YOLO DETECTION ---
        # Run detection on the FAKE thermal image
        results = yolo_model.predict(thermal_bgr, conf=0.6, verbose=0)
        
        # --- STEP C: ANNOTATE ---
        # Draw the boxes on the thermal frame
        annotated_frame = results[0].plot()

        # Write to final video
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"✅ Integration complete! Saved to {output_path}")

# Run it
process_pipeline('test_rgb_video.mp4')