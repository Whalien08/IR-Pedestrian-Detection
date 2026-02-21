import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, BatchNormalization, Dropout # and so on...
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from pathlib import Path
from IPython.display import Image, display

# --- 1. CONFIGURATION ---
CONFIG = {
    "IMG_SIZE": (512, 512, 3), 
    "BATCH_SIZE": 1,
    "EPOCHS": 50,       
    "LR": 2e-4,
    "BETA_1": 0.5,
    "SAVE_FREQ": 1,        # Save every epoch so you don't lose progress
    "DATA_PATH": "C:\Users\Nyx\Desktop\IR Pedestrian Detection\gan\data\training"
}

# --- 2. DATA LOADING & PREPROCESSING ---
def load_and_preprocess(image_path, target_path):
    def process_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1]])
        return (tf.cast(img, tf.float32) / 127.5) - 1.0

    return process_img(target_path), process_img(image_path)

class FastDataset:
    def __init__(self, rgb_list, thermal_list, batch_size=1):
        dataset = tf.data.Dataset.from_tensor_slices((thermal_list, rgb_list))
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.loader = dataset

def get_image_lists(base_path):
    rgb_path = Path(base_path) / "rgb"
    thermal_path = Path(base_path) / "thermal"
    
    # Pathlib globbing
    rgbs = sorted([str(f) for f in rgb_path.glob("*") if f.stat().st_size > 0])
    thermals = sorted([str(f) for f in thermal_path.glob("*") if f.stat().st_size > 0])
    return rgbs, thermals

# --- 3. MODEL ARCHITECTURE ---
def build_generator(input_shape):
    def downsample(filters, size, apply_batchnorm=True):
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same', use_bias=False))
        if apply_batchnorm: result.add(BatchNormalization())
        result.add(LeakyReLU(0.2))
        return result

    def upsample(filters, size, apply_dropout=False):
        result = tf.keras.Sequential()
        result.add(Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False))
        result.add(BatchNormalization())
        if apply_dropout: result.add(Dropout(0.5))
        result.add(Activation('relu'))
        return result

    inputs = Input(shape=input_shape)
    d1 = downsample(64, 4, False)(inputs) 
    d2 = downsample(128, 4)(d1)
    d3 = downsample(256, 4)(d2)
    d4 = downsample(512, 4)(d3)
    d5 = downsample(512, 4)(d4)
    d6 = downsample(512, 4)(d5)
    d7 = downsample(512, 4)(d6)

    u1 = upsample(512, 4, True)(d7)
    u1 = Concatenate()([u1, d6])
    u2 = upsample(512, 4, True)(u1)
    u2 = Concatenate()([u2, d5])
    u3 = upsample(512, 4, True)(u2)
    u3 = Concatenate()([u3, d4])
    u4 = upsample(256, 4)(u3)
    u4 = Concatenate()([u4, d3])
    u5 = upsample(128, 4)(u4)
    u5 = Concatenate()([u5, d2])
    u6 = upsample(64, 4)(u5)
    u6 = Concatenate()([u6, d1])

    last = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(u6)
    return Model(inputs=inputs, outputs=last)

def build_discriminator(input_shape):
    target_img, input_img = Input(shape=input_shape), Input(shape=input_shape)
    combined = Concatenate()([target_img, input_img])
    d1 = LeakyReLU(0.2)(Conv2D(64, 4, strides=2, padding='same')(combined))
    d2 = LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, 4, strides=2, padding='same')(d1)))
    output = Conv2D(1, 4, padding='same')(d2)
    return Model([target_img, input_img], output)

class ThermalGAN:
    def __init__(self, config):
        opt = Adam(config["LR"], beta_1=config["BETA_1"])
        self.generator = build_generator(config["IMG_SIZE"])
        self.discriminator = build_discriminator(config["IMG_SIZE"])
        self.discriminator.compile(loss='mse', optimizer=opt)
        self.discriminator.trainable = False
        
        source_input = Input(shape=config["IMG_SIZE"])
        gen_out = self.generator(source_input)
        validity = self.discriminator([gen_out, source_input])
        self.combined = Model(source_input, [validity, gen_out])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=opt)

# --- 4. TRAINING ENGINE ---
def save_preview(epoch, rgb, thermal, fake):
    rgb, thermal, fake = (rgb[0]+1)/2, (thermal[0]+1)/2, (fake[0]+1)/2
    plt.figure(figsize=(15, 5))
    imgs, titles = [rgb, fake, thermal], ['Input RGB', 'Generated', 'Real Thermal']
    for i in range(3):
        plt.subplot(1, 3, i+1); plt.imshow(imgs[i]); plt.title(titles[i]); plt.axis('off')
    plt.savefig(f'output_visuals/epoch_{epoch+1}.png'); plt.close()

def train(gan_model, dataset, config):
    os.makedirs('output_visuals', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    patch = gan_model.discriminator.output_shape[1]
    valid = np.ones((config["BATCH_SIZE"], patch, patch, 1))
    fake_label = np.zeros((config["BATCH_SIZE"], patch, patch, 1))
    best_g_loss = float('inf')

    for epoch in range(config["EPOCHS"]):
        epoch_g_loss = []
        for batch_i, (real_thermal, real_rgb) in enumerate(dataset.loader):
            fake_thermal = gan_model.generator.predict(real_rgb, verbose=0)
            
            # Train Discriminator
            d_loss_real = gan_model.discriminator.train_on_batch([real_thermal, real_rgb], valid)
            d_loss_fake = gan_model.discriminator.train_on_batch([fake_thermal, real_rgb], fake_label)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator
            g_loss = gan_model.combined.train_on_batch(real_rgb, [valid, real_thermal])
            epoch_g_loss.append(g_loss[0])

            if batch_i % 100 == 0:
                print(f"E{epoch+1} B{batch_i} | D_Loss: {d_loss:.4f} | G_Loss: {g_loss[0]:.4f}")

        # Best Model Saving (Callback Logic)
        avg_g_loss = np.mean(epoch_g_loss)
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            gan_model.generator.save("saved_models/thermal_gen_best.h5")
            print(f" Best Model Updated (Loss: {best_g_loss:.4f})")

        if (epoch + 1) % config["SAVE_FREQ"] == 0:
            save_preview(epoch, real_rgb, real_thermal, fake_thermal)

# --- 5. EXECUTION ---
if __name__ == "__main__":
    rgb_imgs, thermal_imgs = get_image_lists(CONFIG["DATA_PATH"])
    
    if len(rgb_imgs) > 0 and len(rgb_imgs) == len(thermal_imgs):
        print(f" Data Verified: {len(rgb_imgs)} pairs found.")
        gan = ThermalGAN(CONFIG)
        data = FastDataset(rgb_imgs, thermal_imgs, CONFIG["BATCH_SIZE"])
        train(gan, data, CONFIG)
    else:
        print(f"Mismatch: RGB({len(rgb_imgs)}) Thermal({len(thermal_imgs)})")

if os.path.exists('output_visuals'):
    files = sorted(os.listdir('output_visuals'))
    if files:
        print(f"Latest preview: {files[-1]}")
        display(Image(filename=f'output_visuals/{files[-1]}'))
    
if os.path.exists('saved_models'):
    print(f"Models saved: {os.listdir('saved_models')}")  