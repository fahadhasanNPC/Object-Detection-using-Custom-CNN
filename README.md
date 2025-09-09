# Custom CNN on CIFAR-100 Dataset

![CIFAR-100 Banner](https://img.shields.io/badge/Dataset-CIFAR--100-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Overview

This project implements a **Custom Convolutional Neural Network (CNN)** for image classification on the **CIFAR-100** dataset, which consists of 60,000 32x32 color images across 100 fine-grained classes (e.g., apple, bicycle, dolphin). The model is built using TensorFlow and Keras, incorporating techniques like data augmentation, batch normalization, dropout, L2 regularization, and learning rate scheduling to improve performance.

**Key Highlights:**
- **Training Accuracy**: ~56.9%
- **Validation Accuracy**: ~58.9% (best: ~59.9%)
- A simple **GUI** (using Tkinter) for uploading images, displaying predictions, top-3 probabilities, a probability bar chart, and sample images from the dataset.
- Evaluation metrics including confusion matrix, classification report, per-class accuracy, and training/validation curves.

This repository serves as a practical example for building and deploying CNNs for multi-class image classification tasks.

## Features

- **Data Loading & Preprocessing**: Loads CIFAR-100 dataset, normalizes images, and flattens labels.
- **Custom CNN Architecture**:
  - Convolutional layers with ReLU activation and L2 regularization.
  - Batch Normalization and MaxPooling for efficiency.
  - Dropout layers to prevent overfitting.
  - Dense layers for classification (100 classes with softmax).
- **Data Augmentation**: Rotation, shifts, and flips using `ImageDataGenerator`.
- **Training**: Adam optimizer, sparse categorical cross-entropy loss, 30 epochs with learning rate reduction on plateau.
- **Evaluation**:
  - Confusion matrix heatmap.
  - Classification report (precision, recall, F1-score).
  - Per-class accuracy bar chart.
  - Training/validation accuracy and loss plots.
- **GUI Interface**:
  - Upload custom images for prediction.
  - Displays predicted class with a random caption.
  - Top-3 predictions with probabilities and bar chart.
  - Shows sample images from the predicted class.
- **Hyperparameters**: Easily configurable (batch size, epochs, learning rate, etc.).

## Dataset

- **CIFAR-100**: 50,000 training images and 10,000 test images.
- Classes: 100 fine labels (e.g., 'apple', 'aquarium_fish', 'bicycle').
- Source: Loaded via `keras.datasets.cifar100.load_data()`.

## Result 
- The model has been trained till 30 epochs: The accuracy found till 30 epochs: 0.5885 or 58.85%
<img width="1392" height="97" alt="image" src="https://github.com/user-attachments/assets/f01d316c-17c0-438e-ae08-ff6e71a0a4f2" />

## Visualization:
- Confusion Matrix:
<img width="1459" height="1384" alt="Custom Output" src="https://github.com/user-attachments/assets/51d42b34-af30-47cd-96d6-e8aaa94e185f" />

- Training and Testing Graph:
<img width="1005" height="473" alt="Custom Testing" src="https://github.com/user-attachments/assets/cd638566-85fd-4802-9804-0f03310b1880" />


<img width="1005" height="473" alt="Custom Training" src="https://github.com/user-attachments/assets/62d5b228-e25f-49ab-8282-c39f0eca3af1" />


## Comparison Table

| Model Name   | Dataset Used | Accuracy         | Loss    |
|--------------|--------------|------------------|---------|
| Our Model    | CIFAR-100    | 0.5421 (54.21%)  | 2.2151  |
| VGG16        | CIFAR-100    | 0.01             | 4.6052  |
| ResNet       | CIFAR-100    | 0.5167           | 3.6816  |
| MobileNetV2  | CIFAR-100    | 0.154            | 4.7598  |

- **Optimizer**: Adam
- **Hyperparameters**: 30 epochs, categorical crossentropy, batch size=64, image size differed across models.



## GUI Interface:
<img width="1920" height="1080" alt="Screenshot 2025-08-27 201235" src="https://github.com/user-attachments/assets/9ad8d7c7-6a0a-4efb-b013-58de7ab4508c" />


