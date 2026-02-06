import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

print("=" * 50)
print("TRAFFIC SIGN PREDICTION TOOL")
print("=" * 50)

# Load the trained model
model = keras.models.load_model('best_model.h5')
print("Model loaded successfully\n")


# Load class names (optional - for better display)
# If you have a file mapping ClassId to sign names, load it here
# For now, we'll just use class numbers

def predict_traffic_sign(image_path):
    """
    Predict the traffic sign in the given image
    """
    IMG_SIZE = 32

    # Load and preprocess the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Make prediction
    predictions = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100

    # Display results
    plt.figure(figsize=(10, 5))

    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Show preprocessed image with prediction
    plt.subplot(1, 2, 2)
    plt.imshow(img_resized)
    plt.title(f'Prediction: Class {predicted_class}\nConfidence: {confidence:.2f}%')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'prediction_result_class_{predicted_class}.png')
    plt.show()

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    # Show top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    print("\nTop 5 Predictions:")
    for i, idx in enumerate(top_5_indices):
        print(f"{i + 1}. Class {idx}: {predictions[0][idx] * 100:.2f}%")

    return predicted_class, confidence


# Example usage
print("Usage:")
print("predict_traffic_sign('path/to/your/image.png')")
print("\nExample with test image:")

# Test with a random image from test set
test_df = pd.read_csv('./gtsrb_data/Test.csv')
sample_image = test_df.iloc[0]
sample_path = './gtsrb_data/' + sample_image['Path']

print(f"\nTesting with: {sample_path}")
print(f"True label: {sample_image['ClassId']}\n")

predicted_class, confidence = predict_traffic_sign(sample_path)

print("\n" + "=" * 50)
print("You can now use this function to predict any traffic sign image!")
print("=" * 50)