import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("=" * 50)
print("TESTING THE MODEL")
print("=" * 50)

# Load the best trained model
print("\nLoading trained model...")
model = keras.models.load_model('best_model.h5')
print("Model loaded successfully")

# Load test data
print("\nLoading test data...")
test_df = pd.read_csv('./gtsrb_data/Test.csv')
print(f"Total test images: {len(test_df)}")

# Preprocess test images
IMG_SIZE = 32


def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img


print("\nPreprocessing test images...")
X_test = []
y_test = []

for idx, row in test_df.iterrows():
    img_path = './gtsrb_data/' + row['Path']
    label = row['ClassId']

    img = load_and_preprocess_image(img_path)
    X_test.append(img)
    y_test.append(label)

    if (idx + 1) % 2000 == 0:
        print(f"Processed {idx + 1}/{len(test_df)} images...")

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

print(f"\nTest data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Evaluate the model
print("\n" + "=" * 50)
print("EVALUATING MODEL ON TEST DATA")
print("=" * 50)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions
print("\nMaking predictions on test set...")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Generate classification report
print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_test, predicted_classes, target_names=[str(i) for i in range(43)]))

# Calculate per-class accuracy
print("\nPer-Class Accuracy:")
for class_id in range(43):
    class_mask = y_test == class_id
    if np.sum(class_mask) > 0:
        class_accuracy = np.mean(predicted_classes[class_mask] == y_test[class_mask])
        print(f"Class {class_id}: {class_accuracy:.4f} ({class_accuracy * 100:.2f}%)")

# Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix - Traffic Sign Classification')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.close()

# Show some correct and incorrect predictions
print("\nGenerating prediction samples...")

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Sample Predictions', fontsize=16)

# Get some random samples
sample_indices = np.random.choice(len(X_test), 20, replace=False)

for i, ax in enumerate(axes.flat):
    idx = sample_indices[i]

    ax.imshow(X_test[idx])

    true_label = y_test[idx]
    pred_label = predicted_classes[idx]
    confidence = predictions[idx][pred_label] * 100

    if true_label == pred_label:
        color = 'green'
        result = 'CORRECT'
    else:
        color = 'red'
        result = 'WRONG'

    ax.set_title(f'{result}\nTrue: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%',
                 color=color, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('prediction_samples.png', dpi=150)
print("Prediction samples saved as 'prediction_samples.png'")
plt.close()

# Find misclassified images
misclassified_indices = np.where(predicted_classes != y_test)[0]
print(f"\nTotal misclassified images: {len(misclassified_indices)} out of {len(y_test)}")

if len(misclassified_indices) > 0:
    print("\nShowing some misclassified examples...")

    num_errors_to_show = min(10, len(misclassified_indices))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Misclassified Examples', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_errors_to_show:
            idx = misclassified_indices[i]

            ax.imshow(X_test[idx])

            true_label = y_test[idx]
            pred_label = predicted_classes[idx]
            confidence = predictions[idx][pred_label] * 100

            ax.set_title(f'True: {true_label} | Predicted: {pred_label}\nConfidence: {confidence:.1f}%',
                         fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=150)
    print("Misclassified examples saved as 'misclassified_examples.png'")
    plt.close()

print("\n" + "=" * 50)
print("TESTING COMPLETE")
print("=" * 50)
print("\nGenerated files:")
print("- confusion_matrix.png: Shows which classes get confused")
print("- prediction_samples.png: Random predictions with labels")
print("- misclassified_examples.png: Examples of errors")