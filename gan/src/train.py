import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D, LeakyReLU, 
    Activation, Concatenate, BatchNormalization, Dropout, RandomRotation
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from pathlib import Path
import random

# 1. CONFIGURATION 
CONFIG = {
    "IMG_SIZE": (512, 512, 3),
    "BATCH_SIZE": 1,
    "EPOCHS": 22,
    "G_LR": 2e-4,      
    "D_LR": 5e-5,      # Lower D_LR to prevent collapse
    "BETA_1": 0.5,
    "SAVE_FREQ": 1,
    "DATA_PATH": "/kaggle/input/datasets/himawariricttoslock/pedestrian-thermal-data/data"
}

# 2. DATA LOADING & PREPROCESSING 
# Instantiate RandomRotation outside the mapped function to prevent variable creation in graph mode
rotation_layer = RandomRotation(factor=0.02)

def load_and_preprocess(image_path, target_path):
    def process_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [542, 542], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return img

    t_img = process_img(target_path)
    r_img = process_img(image_path)
    
    # Concatenate along channel axis to ensure identical spatial transformations
    concat_img = tf.concat([t_img, r_img], axis=-1)
    
    # Random Crop
    cropped = tf.image.random_crop(concat_img, size=[512, 512, 6])
    
    # Random Horizontal Flip (handles probability internally)
    cropped = tf.image.random_flip_left_right(cropped)
        
    # Expand dims for RandomRotation, rotate, and squeeze
    cropped = tf.expand_dims(cropped, axis=0)
    rotated = rotation_layer(cropped, training=True)
    rotated = tf.squeeze(rotated, axis=0)
    
    t_img_aug = rotated[..., :3]
    r_img_aug = rotated[..., 3:]
    
    t_final = (tf.cast(t_img_aug, tf.float32) / 127.5) - 1.0
    r_final = (tf.cast(r_img_aug, tf.float32) / 127.5) - 1.0
    return r_final, t_final

class FastDataset:
    def __init__(self, rgb_list, thermal_list, batch_size=1):
        dataset = tf.data.Dataset.from_tensor_slices((rgb_list, thermal_list))
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.loader = dataset

def get_image_lists(base_path):
    rgb_path = Path(base_path) / "rgb"
    thermal_path = Path(base_path) / "thermal"
    rgbs = sorted([str(f) for f in rgb_path.glob("*") if f.stat().st_size > 0])
    thermals = sorted([str(f) for f in thermal_path.glob("*") if f.stat().st_size > 0])
    return rgbs, thermals

# 3. MODEL ARCHITECTURE 
def build_generator(input_shape):
    def downsample(filters, size, apply_batchnorm=True):
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same', use_bias=False))
        if apply_batchnorm: result.add(BatchNormalization())
        result.add(LeakyReLU(0.2))
        return result

    def upsample(filters, size, apply_dropout=False):
        result = tf.keras.Sequential()
        result.add(UpSampling2D(size=(2, 2)))
        result.add(Conv2D(filters, size, padding='same', use_bias=False))
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

    last = UpSampling2D(size=(2, 2))(u6)
    last = Conv2D(3, 4, padding='same', activation='tanh')(last)
    return Model(inputs=inputs, outputs=last)

def build_discriminator(input_shape):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        if bn: d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    img_A = Input(shape=input_shape) # Target
    img_B = Input(shape=input_shape) # Input
    combined = Concatenate(axis=-1)([img_A, img_B])
    d1 = d_layer(combined, 64, bn=False)
    d2 = d_layer(d1, 128)
    d3 = d_layer(d2, 256)
    d4 = d_layer(d3, 512)
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    return Model([img_A, img_B], validity)


def build_vgg_feature_extractor(input_shape):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg.trainable = False
    # Using block3_conv3 as requested
    output_layer = vgg.get_layer('block3_conv3').output
    model = Model(inputs=vgg.input, outputs=output_layer)
    return model

class ThermalGAN:
    def __init__(self, config):
        self.generator = build_generator(config["IMG_SIZE"])
        self.discriminator = build_discriminator(config["IMG_SIZE"])
        self.vgg = build_vgg_feature_extractor(config["IMG_SIZE"])
        self.g_opt = Adam(config["G_LR"], beta_1=config["BETA_1"])
        self.d_opt = Adam(config["D_LR"], beta_1=config["BETA_1"])

#  4. TRAINING ENGINE 
@tf.function
def train_step(gan, real_rgb, real_thermal, valid, fake_label):
    smoothed_valid = valid * 0.9 
    
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_thermal = gan.generator(real_rgb, training=True)
        
        pred_real = gan.discriminator([real_thermal, real_rgb], training=True)
        pred_fake = gan.discriminator([fake_thermal, real_rgb], training=True)
        
        d_loss = 0.5 * (tf.reduce_mean(tf.square(smoothed_valid - pred_real)) + 
                        tf.reduce_mean(tf.square(fake_label - pred_fake)))
        
        # Generator Losses
        validity = gan.discriminator([fake_thermal, real_rgb], training=False)
        g_loss_gan = tf.reduce_mean(tf.square(valid - validity))
        g_loss_l1 = tf.reduce_mean(tf.abs(real_thermal - fake_thermal))
        
        # VGG Perceptual Loss (scale from [-1, 1] to [0, 255])
        real_thermal_vgg = preprocess_input((real_thermal + 1.0) * 127.5)
        fake_thermal_vgg = preprocess_input((fake_thermal + 1.0) * 127.5)
        
        real_features = gan.vgg(real_thermal_vgg, training=False)
        fake_features = gan.vgg(fake_thermal_vgg, training=False)
        g_loss_perceptual = tf.reduce_mean(tf.abs(real_features - fake_features))
        
        total_g_loss = g_loss_gan + (50 * g_loss_l1) + (10 * g_loss_perceptual)

    g_grads = g_tape.gradient(total_g_loss, gan.generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss, gan.discriminator.trainable_variables)
    
    gan.g_opt.apply_gradients(zip(g_grads, gan.generator.trainable_variables))
    gan.d_opt.apply_gradients(zip(d_grads, gan.discriminator.trainable_variables))
    return d_loss, g_loss_gan, g_loss_l1, g_loss_perceptual, total_g_loss, fake_thermal

def train(gan_model, dataset, config, start_epoch=0):
    os.makedirs('output_visuals', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    csv_file = 'training_losses.csv'
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Epoch', 'D_Loss', 'G_Loss_GAN', 'G_Loss_L1', 'G_Loss_Perceptual', 'Total_G_Loss'])
    
    patch = gan_model.discriminator.output_shape[1]
    valid = tf.ones((config["BATCH_SIZE"], patch, patch, 1))
    fake_label = tf.zeros((config["BATCH_SIZE"], patch, patch, 1))

    for epoch in range(start_epoch, config["EPOCHS"]):
        epoch_d_loss = []
        epoch_g_gan = []
        epoch_g_l1 = []
        epoch_g_perceptual = []
        epoch_total_g = []
        
        for batch_i, (real_rgb, real_thermal) in enumerate(dataset.loader):
            d_loss, g_loss_gan, g_loss_l1, g_loss_perceptual, total_g_loss, fake_thermal = train_step(gan_model, real_rgb, real_thermal, valid, fake_label)
            
            epoch_d_loss.append(d_loss)
            epoch_g_gan.append(g_loss_gan)
            epoch_g_l1.append(g_loss_l1)
            epoch_g_perceptual.append(g_loss_perceptual)
            epoch_total_g.append(total_g_loss)
            
            if batch_i % 100 == 0:
                print(f"E{epoch+1} B{batch_i} | D: {d_loss:.4f} | G_GAN: {g_loss_gan:.4f} | G_L1: {g_loss_l1:.4f} | G_VGG: {g_loss_perceptual:.4f} | G_Tot: {total_g_loss:.4f}")

        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_gan = np.mean(epoch_g_gan)
        avg_g_l1 = np.mean(epoch_g_l1)
        avg_g_perceptual = np.mean(epoch_g_perceptual)
        avg_total_g = np.mean(epoch_total_g)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_d_loss, avg_g_gan, avg_g_l1, avg_g_perceptual, avg_total_g])
        
        print(f"--> Epoch {epoch+1} Averages | D: {avg_d_loss:.4f} | G_GAN: {avg_g_gan:.4f} | G_L1: {avg_g_l1:.4f} | G_VGG: {avg_g_perceptual:.4f} | G_Tot: {avg_total_g:.4f}")

        if (epoch + 1) % config["SAVE_FREQ"] == 0:
            save_preview(epoch, real_rgb, real_thermal, fake_thermal)
            gan_model.generator.save(f"saved_models/thermal_gen_epoch_{epoch+1}.keras")

def save_preview(epoch, rgb, thermal, fake):
    rgb, thermal, fake = (rgb[0]+1)/2, (thermal[0]+1)/2, (fake[0]+1)/2
    plt.figure(figsize=(15, 5))
    imgs, titles = [rgb, fake, thermal], ['Input RGB', 'Generated', 'Real Thermal']
    for i in range(3):
        plt.subplot(1, 3, i+1); plt.imshow(imgs[i]); plt.title(titles[i]); plt.axis('off')
    plt.savefig(f'output_visuals/epoch_{epoch+1}.png'); plt.close()


# 5. EXECUTION 
if __name__ == "__main__":
    rgb_imgs, thermal_imgs = get_image_lists(CONFIG["DATA_PATH"])
    if len(rgb_imgs) > 0 and len(rgb_imgs) == len(thermal_imgs):
        gan = ThermalGAN(CONFIG)
        
        # RESUME FROM EPOCH 12
        weight_path = '/kaggle/input/notebooks/himawariricttoslock/gan-4-3/saved_models/thermal_gen_epoch_12.h5'
        
        # --- THE FIX: Initialize start_epoch before the if statement ---
        start_epoch = 0 
        
        if os.path.exists(weight_path):
            gan.generator.load_weights(weight_path)
            # Set to 12 so the loop starts at 13
            start_epoch = 12 
            # Set target to 22 (approx 2 hours of work)
            CONFIG["EPOCHS"] = 22 
            print(f"✅ Resuming from Epoch 12. Target: Epoch 22.")
        else:
            print(f"⚠️ WARNING: Could not find weights at {weight_path}")
            print(f"⚠️ Make sure your 'gan-4-3' notebook is attached in the right-hand data panel! Starting from Epoch 0.")
        
        data = FastDataset(rgb_imgs, thermal_imgs, CONFIG["BATCH_SIZE"])
        train(gan, data, CONFIG, start_epoch=start_epoch)