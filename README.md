
**Brain Tumor Detection Using CNN**

This repository contains code and a deep learning model to detect brain tumors from MRI images using Convolutional Neural Networks (CNN). The project leverages MobileNet as the base model, fine-tuned to classify MRI images into two classes: brain tumor positive and healthy.

Introduction
Brain tumor detection is a critical medical imaging task, where early and accurate diagnosis is essential for treatment planning and patient outcomes. In this project, a deep learning model is trained on MRI images to detect brain tumors, achieving high accuracy through data augmentation and fine-tuning of a pre-trained MobileNet model.

Dataset
The dataset consists of MRI images labeled as either:

Positive (with brain tumor)
Negative (healthy)
Dataset Split:
Training Set: 70%
Validation Set: 15%
Test Set: 15%


Model Architecture
The model is based on MobileNet, a pre-trained CNN model. The key components are:

MobileNet Base Model: Extracts image features.
Fully Connected Layer: Fine-tuned layers to classify images.
Dropout Layers: Added to prevent overfitting.
Binary Output Layer: Outputs either 0 (negative) or 1 (positive) for brain tumor classification.

Data Augmentation:
Data augmentation techniques like zoom, shear, and horizontal flips are applied to increase the diversity of training data and improve model generalization.

Key Training Features:
Callbacks: ModelCheckpoint and EarlyStopping to save the best model and stop training when no further improvement is observed.

Evaluation
Once training is complete, the model can be evaluated on the test set:
The model achieves an accuracy of 95.76% on the test set.

Results
Training Accuracy: 95.52%
Validation Accuracy: 96.68%
Test Accuracy: 95.76%
Loss and accuracy graphs during training are plotted to monitor the model’s performance and prevent overfitting.

Usage
You can use the trained model to classify new MRI images. 
