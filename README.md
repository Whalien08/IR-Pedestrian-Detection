# Infrared Pedestrian Detection 

## Overview

This project focuses on enhancing pedestrian detection in low-visibility conditions by generating synthetic infrared (IR) images from RGB data using a Generative Adversarial Network (GAN). These synthetic images are used to train and validate a YOLO + FPN detection model, allowing for robust perception in 3D traffic simulations.

## Technical Workflow

    1. Dataset Preparation: Resizing and normalizing 10,400 paired RGB-Thermal images.

    2. Synthetic Generation: Training a GAN (Epoch 22 selected as optimal) to translate RGB features into thermal  heat signatures.

    3. Perception: Using YOLO to detect pedestrians in generated thermal frames, specifically handling overlapping subjects.

    4. Integration: Visualizing detections within a simulated traffic environment.

## Data Flow Diagram

<img width="2818" height="1024" alt="Screenshot 2026-02-27 201952" src="https://github.com/user-attachments/assets/720cdec0-80e6-4cda-aaff-ecde60a1d7ab" />



## Technical Implementation: GAN Architecture

The project utilizes a Pix2Pix-style Generative Adversarial Network designed for image-to-image translation (RGB → Infrared).

## GAN Structure

<img width="1200" height="540" alt="structure_of_a_gan" src="https://github.com/user-attachments/assets/18acb0d4-fc43-45bc-8296-76897c701b5c" />


### 1. Model Architecture

The system consists of a U-Net Generator for high-resolution feature mapping and a PatchGAN Discriminator to ensure local texture realism.

    Generator: Employs a symmetric encoder-decoder structure with skip connections to preserve spatial details from the input RGB frames.

    Discriminator: A Markovian discriminator (PatchGAN) that penalizes structure at the scale of image patches, promoting sharper thermal gradients.

### 2. Training Engine & Optimization

The training process was stabilized using Two-Time-Scale Update Rule (TTUR) and One-Sided Label Smoothing to prevent mode collapse.

### Stabilization Strategy in train_step

    @tf.function
    def train_step(gan, real_rgb, real_thermal, valid, fake_label):
        # One-Sided Label Smoothing to keep the Discriminator from overpowering the Generator
        smoothed_valid = valid * 0.9 
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # Generate synthetic thermal images
            fake_thermal = gan.generator(real_rgb, training=True)
            
            # Discriminator evaluation
            pred_real = gan.discriminator([real_thermal, real_rgb], training=True)
            pred_fake = gan.discriminator([fake_thermal, real_rgb], training=True)
            
            # LSGAN Loss calculation
            d_loss = 0.5 * (tf.reduce_mean(tf.square(smoothed_valid - pred_real)) + 
                            tf.reduce_mean(tf.square(fake_label - pred_fake)))
            
            # Generator Loss (GAN Loss + L1 Pixel-wise Reconstruction Loss)
            validity = gan.discriminator([fake_thermal, real_rgb], training=False)
            g_loss_gan = tf.reduce_mean(tf.square(valid - validity))
            g_loss_l1 = tf.reduce_mean(tf.abs(real_thermal - fake_thermal))
            total_g_loss = g_loss_gan + (100 * g_loss_l1)

        # Gradient Application
        g_grads = g_tape.gradient(total_g_loss, gan.generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, gan.discriminator.trainable_variables)
        
        gan.g_opt.apply_gradients(zip(g_grads, gan.generator.trainable_variables))
        gan.d_opt.apply_gradients(zip(d_grads, gan.discriminator.trainable_variables))
        return d_loss, total_g_loss, fake_thermal

### 3. Configuration & Hyperparameters

To achieve optimal results at Epoch 22, the following parameters were utilized:

    Input Size: 512x512x3

    Generator Learning Rate (GLR​): 2×10−4

    Discriminator Learning Rate (DLR​): 5×10−5 (Slower to maintain stability)

    Optimizer: Adam (β1​=0.5)

    L1 Weight: 100 (High importance on structural accuracy)

###RGB Image
<img width="559" height="536" alt="image" src="https://github.com/user-attachments/assets/1b247f0a-1740-4db5-ad2b-41b70541129c" />

### Synthetic Image
<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/edc15b34-54c1-4efc-8d4c-60d3abb3b19d" />



## Model Architecture: YOLOv8

This project utilizes the YOLOv8 architecture for real-time object detection. YOLOv8 is a state-of-the-art model that unifies feature extraction, multi-scale fusion, and decoupled detection heads.



### **Detection Architecture (YOLOv8)**
The project utilizes a YOLOv8 framework for real-time pedestrian perception.

## YOLOv8 Architecture 

<img width="850" height="717" alt="YOLOv8-overall-architecture-diagram" src="https://github.com/user-attachments/assets/34e1ad35-1a29-4aa8-84f5-ec444d21061e" />


## 1. Backbone (Feature Extractor)

The backbone is responsible for extracting hierarchical features from the input image.

    CSPDarknet53: Used to capture both low-level textures and high-level semantic information.

    C2f Module: Replaces the C3 modules from previous versions. It uses additional skip connections and split operations to improve gradient flow and learning efficiency.

    SPPF (Spatial Pyramid Pooling-Fast): Pools feature maps into a fixed size, ensuring the model handles various input dimensions effectively.

## 2. Neck (Feature Aggregator)

The neck acts as a bridge, fusing multi-scale features to detect objects of different sizes.

    PAN-FPN: YOLOv8 employs a Path Aggregation Network (PAN) combined with a Feature Pyramid Network (FPN) structure.

    Function: It aggregates feature maps from different stages of the backbone, enabling the model to retain both spatial detail and semantic context.

## 3. Head (Detection)

The detection head generates the final bounding boxes and class probabilities.

    Anchor-Free Design: Unlike older versions, YOLOv8 predicts an object's midpoint directly rather than using predefined anchor boxes.

    Decoupled Head: The classification and regression (bounding box) tasks are processed in separate branches, which significantly improves detection performance.



## Results

The final model generates IR images where pedestrians are clearly distinguishable from the background, even when overlapping. This allows for more reliable pedestrian detection in simulated traffic scenarios.
