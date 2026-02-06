import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import os

print("=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

# Configuration
IMG_SIZE = 32
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Load training data CSV
train_df = pd.read_csv('./gtsrb_data/Train.csv')

print(f"\nLoading and preprocessing {len(train_df)} images...")


# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Convert from BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to standard size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values to range [0, 1]
    img = img / 255.0

    return img


# Load all images and labels
X_data = []
y_data = []

for idx, row in train_df.iterrows():
    img_path = './gtsrb_data/' + row['Path']
    label = row['ClassId']

    # Load and preprocess image
    img = load_and_preprocess_image(img_path)

    X_data.append(img)
    y_data.append(label)

    # Progress indicator
    if (idx + 1) % 5000 == 0:
        print(f"Processed {idx + 1}/{len(train_df)} images...")

# Convert lists to numpy arrays
X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.int32)

print(f"\nData shape: {X_data.shape}")
print(f"Labels shape: {y_data.shape}")
print(f"Data type: {X_data.dtype}")
print(f"Value range: [{X_data.min():.2f}, {X_data.max():.2f}]")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_data,
    y_data,
    test_size=VALIDATION_SPLIT,
    random_state=42,
    stratify=y_data
)

print(f"\nTraining set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")

# Save preprocessed data
print("\nSaving preprocessed data...")
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

print("\nPreprocessing complete!")
print("=" * 50)
print("\nSaved files:")
print("- X_train.npy: Training images")
print("- X_val.npy: Validation images")
print("- y_train.npy: Training labels")
print("- y_val.npy: Validation labels")