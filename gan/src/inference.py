import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the model
def load_model(model_path):
    """ Load the TensorFlow model from the specified path. """
    try:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model

# Preprocess the input image
def preprocess_image(img, target_size=(512, 512)):
    """ Resize and normalize the image for the model input. """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    return (img_resized / 127.5) - 1.0

# Post-process the predicted thermal data
def postprocess_thermal_data(prediction):
    # 1. Basic Denormalization
    thermal_data = np.mean(prediction[0], axis=-1)
    thermal_data = ((thermal_data + 1.0) * 127.5).astype(np.uint8)

    # 2. gentle contrast (CLAHE is better than equalizeHist)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    thermal_data = clahe.apply(thermal_data)

    # 3. Clean up the "GAN mesh" noise
    thermal_data = cv2.medianBlur(thermal_data, 3)
    # Add this right before the 'return' statement
    gaussian_3 = cv2.GaussianBlur(thermal_data, (0, 0), 2.0)
    thermal_data = cv2.addWeighted(thermal_data, 1.5, gaussian_3, -0.5, 0)

    return thermal_data



# Display the thermal results
def display_thermal_results(input_rgb_folder, model, num_to_show=8):
    """ Process images to generate and display thermal results. """
    output_dir = '/kaggle/working/thermal_outputs'
    os.makedirs(output_dir, exist_ok=True)

    img_list = [f for f in os.listdir(input_rgb_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_to_show]
    
    cols = 2
    rows = (len(img_list) + 1) // 2
    plt.figure(figsize=(15, rows * 5))

    for i, img_name in enumerate(tqdm(img_list)):
        path = os.path.join(input_rgb_folder, img_name)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Couldn't read image {img_name}")
            continue
        
        # Preprocessing
        input_arr = preprocess_image(img)
        
        # Inference
        prediction = model.predict(np.expand_dims(input_arr, 0), verbose=0)
        
        # Post-processing
        thermal_data = postprocess_thermal_data(prediction)

        # Save output
        save_path = os.path.join(output_dir, f"thermal_{img_name}")
        cv2.imwrite(save_path, thermal_data)

        # Plotting
        plt.subplot(rows, cols, i + 1)
        plt.imshow(thermal_data, cmap='gray') 
        plt.title(f"Thermal: {img_name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    print(f"Images saved successfully to: {output_dir}")

# Run the script
if __name__ == "__main__":
    model_path = '/kaggle/input/notebooks/himawariricttoslock/gan-4-3/saved_models/thermal_gen_epoch_22.h5'
    model = load_model(model_path)

    if model:
        display_thermal_results('/kaggle/input/datasets/himawariricttoslock/input-170s/rgb images', model, num_to_show=170)
