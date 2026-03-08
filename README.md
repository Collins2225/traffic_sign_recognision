Project Overview

This project focuses on building a computer vision model capable of recognizing and classifying traffic signs from images. 
Traffic sign recognition is a critical component in autonomous driving systems and intelligent transportation technologies, enabling vehicles to understand road rules and respond accordingly.

A Convolutional Neural Network (CNN) is trained on a dataset of traffic sign images to automatically identify and classify different signs such as Stop, Yield, Speed Limit, and others.


Objectives

.The main goals of this project are to:

.Develop a deep learning model for traffic sign classification

.Preprocess and prepare image data for training

.Train a CNN-based image classification model

.Evaluate the model’s performance using accuracy and validation metrics

.Demonstrate how computer vision can be applied in autonomous vehicle perception systems


Methodology

.The project follows a standard computer vision pipeline:

.Data Collection

.Traffic sign images obtained from a public dataset (e.g., German Traffic Sign Recognition Benchmark).

.Data Preprocessing

.Image resizing

.Normalization

.Data augmentation (optional)

.Train/validation/test split

.Model Development

.Build a Convolutional Neural Network (CNN)

.Train the model on labeled traffic sign images

.Model Evaluation

.Evaluate performance using:

.Accuracy

.Loss curves

.Confusion matrix

.Prediction

.The trained model predicts the class of unseen traffic sign images.


Model Output

.The trained CNN model can correctly classify traffic signs such as:

.Stop Sign

.Yield Sign

.Speed Limit Signs

.No Entry

.Warning Signs


Project Structure

traffic-sign-recognition/
│
├── dataset/              # Traffic sign image dataset
├── preprocessing.py      # Image preprocessing scripts
├── train_model.py        # CNN training script
├── evaluate_model.py     # Model evaluation
├── predict.py            # Predict traffic signs from new images
├── model/                # Saved trained model
└── README.md

How to Run the Project
1.Clone the repository
git clone https://github.com/yourusername/traffic-sign-recognition.git

2.Install required libraries
pip install tensorflow opencv-python numpy matplotlib scikit-learn

3.Train the model
python train_model.py

4.Test predictions.
 python predict.py

Results
The CNN model is able to learn visual patterns in traffic signs and classify them with high accuracy.
This demonstrates the effectiveness of deep learning for visual perception tasks in intelligent transportation systems.








