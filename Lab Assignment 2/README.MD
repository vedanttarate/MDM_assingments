# Lab Assignment 2 - Transfer Learning (ResNet50 on CIFAR-10)

## Overview

This assignment implements transfer learning for image classification using a pre-trained ResNet50 model. The workflow follows a research paper approach and adapts it to the CIFAR-10 dataset.

## Research Paper

- Title: Transfer Learning Using Deep Convolutional Neural Networks for Image Classification
- Model: ResNet50
- Technique: Transfer Learning + Fine-Tuning

## Dataset

- CIFAR-10 (60,000 images, 10 classes)
- Images are normalized and resized from 32x32 to 128x128 for training
- Split: train / validation / test

## Workflow Summary

1. Load and preprocess CIFAR-10
2. Resize and normalize images
3. Build ResNet50 backbone with custom classification head
4. Freeze base layers and train top layers
5. Evaluate with accuracy, classification report, and confusion matrix
6. Plot accuracy and loss curves

## Model Architecture

- Base: ResNet50 (ImageNet weights, include_top=False)
- Head:
  - GlobalAveragePooling2D
  - Dense(128, relu)
  - Dropout(0.5)
  - Dense(10, softmax)

## Results (Summary)

- Accuracy: ~85-90% (approximate for short training run)
- Metrics: precision, recall, F1-score from classification report

## Suggested Improvements

- Train for more epochs (10-20)
- Unfreeze last N layers and fine-tune
- Add data augmentation
- Try EfficientNet or other backbones

## How To Run

Open the notebook and run the cells in order:

1. Dataset preparation
2. Model definition
3. Training
4. Evaluation and plots

Notebook: Transfer_Learning.ipynb

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Author

Rohit Vishwas Thorat
