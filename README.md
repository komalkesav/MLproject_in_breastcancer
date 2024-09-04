# MLproject_in_breastcancer
# Breast Cancer Detection using Deep Learning Models

This repository contains the implementation of multiple deep learning models for breast cancer detection. The models used include VGG16, Vision Transformer (ViT), and ResNet. The objective is to compare the performance of these models on mammography images to accurately classify them as cancerous or non-cancerous.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Models Implemented](#models-implemented)
3. [Dataset](#dataset)

## Project Overview

Breast cancer is one of the leading causes of death among women worldwide. Early detection through accurate classification of mammography images can significantly improve survival rates. This project explores the application of various state-of-the-art deep learning models to enhance the accuracy of breast cancer detection.

## Models Implemented

The following models are implemented and compared:

- **VGG16**: A convolutional neural network (CNN) model known for its deep architecture with small convolutional filters.
- **Vision Transformer (ViT)**: A transformer-based model that leverages self-attention mechanisms, originally developed for NLP, and adapted for image classification.
- **ResNet (Residual Networks)**: A CNN model with skip connections, allowing for the training of very deep networks by mitigating the vanishing gradient problem.
- **custom**: A convolutional neural network (CNN) model using resnet model weights.

## Dataset

The dataset used for training and evaluation is sourced from the [RSNA Breast Cancer Detection Challenge](https://www.kaggle.com/c/rsna-breast-cancer-detection) on Kaggle. It consists of mammography images in DICOM format, labeled as either cancerous or non-cancerous.
