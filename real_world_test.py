import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

print("=" * 50)
print("REAL-WORLD TRAFFIC SIGN TESTING")
print("=" * 50)

# Load the trained model
model = keras.models.load_model('best_model.h5')
print("Model loaded successfully\n")

# Class ID to Sign Name mapping (based on GTSRB dataset)
class_names = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}


def predict_traffic_sign(image_path):
    """
    Predict the traffic sign in the given image
    """
    IMG_SIZE = 32

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return None, None

    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image - {image_path}")
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Make prediction
    predictions = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100

    # Get sign name
    sign_name = class_names.get(predicted_class, f"Unknown (Class {predicted_class})")

    # Display results
    plt.figure(figsize=(12, 5))

    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image', fontsize=14)
    plt.axis('off')

    # Show preprocessed image with prediction
    plt.subplot(1, 2, 2)
    plt.imshow(img_resized)
    title_text = f'Prediction: {sign_name}\nClass {predicted_class}\nConfidence: {confidence:.2f}%'
    plt.title(title_text, fontsize=12)
    plt.axis('off')

    plt.tight_layout()

    # Save result
    filename = os.path.basename(image_path)
    output_name = f'result_{filename}'
    plt.savefig(output_name)
    print(f"Result saved as: {output_name}")
    plt.show()

    # Print results
    print(f"\nPredicted Sign: {sign_name}")
    print(f"Class ID: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    # Show top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    print("\nTop 5 Predictions:")
    for i, idx in enumerate(top_5_indices):
        sign = class_names.get(idx, f"Class {idx}")
        print(f"{i + 1}. {sign}: {predictions[0][idx] * 100:.2f}%")

    print("=" * 50)

    return predicted_class, confidence


def test_multiple_images(image_folder):
    """
    Test all images in a folder
    """
    if not os.path.exists(image_folder):
        print(f"Error: Folder not found - {image_folder}")
        return

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_folder)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(image_files)} images to test\n")

    results = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        print(f"\nTesting: {img_file}")
        print("-" * 50)
        pred_class, confidence = predict_traffic_sign(img_path)
        if pred_class is not None:
            results.append({
                'file': img_file,
                'class': pred_class,
                'sign': class_names.get(pred_class, f"Class {pred_class}"),
                'confidence': confidence
            })

    # Summary
    print("\n" + "=" * 50)
    print("TESTING SUMMARY")
    print("=" * 50)
    for result in results:
        print(f"{result['file']}: {result['sign']} ({result['confidence']:.2f}%)")


# Instructions
print("\nHow to use this script:")
print("\n1. Test a single image:")
print("   predict_traffic_sign('path/to/your/image.jpg')")
print("\n2. Test multiple images in a folder:")
print("   test_multiple_images('path/to/folder')")
print("\n" + "=" * 50)
# Test all your downloaded images
print("\nTesting your downloaded German traffic signs...")
print("="*50)

# Test each image individually
predict_traffic_sign('./German_Traffick_Sign/image 1.jpg')
predict_traffic_sign('./German_Traffick_Sign/image 2.jpg')
predict_traffic_sign('./German_Traffick_Sign/image 3.jpg')
predict_traffic_sign('./German_Traffick_Sign/image 4.jpg')