import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Configure matplotlib to not block execution
plt.ion()

# ============================================================
# SECTION 1: EXTRACT DATASET
# ============================================================

zip_path = r"C:\Users\Collins\Downloads\traffic_sign_dataset.zip"
extract_path = "./gtsrb_data"

os.makedirs(extract_path, exist_ok=True)

print("Extracting dataset... This may take a minute.")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Extraction complete!")

print("\n" + "=" * 50)
print("FOLDER STRUCTURE:")
print("=" * 50)

for root, dirs, files in os.walk(extract_path):
    level = root.replace(extract_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')

    if level < 2:
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files) - 5} more files')

# ============================================================
# SECTION 2: EXPLORE THE DATASET
# ============================================================

print("\n" + "=" * 50)
print("EXPLORING THE DATASET")
print("=" * 50)

train_df = pd.read_csv('./gtsrb_data/Train.csv')

print(f"\nTotal training images: {len(train_df)}")
print(f"Number of classes: {train_df['ClassId'].nunique()}")
print(f"\nFirst few rows of training data:")
print(train_df.head())

print(f"\nImages per class:")
class_counts = train_df['ClassId'].value_counts().sort_index()
print(class_counts)

plt.figure(figsize=(15, 5))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Class ID (Traffic Sign Type)')
plt.ylabel('Number of Images')
plt.title('Distribution of Training Images Across Classes')
plt.xticks(range(43))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('class_distribution.png')
print("\nClass distribution chart saved as 'class_distribution.png'")
plt.close()

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle('Sample Traffic Signs from Different Classes', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < 15:
        class_images = train_df[train_df['ClassId'] == i]
        if len(class_images) > 0:
            sample_path = './gtsrb_data/' + class_images.iloc[0]['Path']
            img = Image.open(sample_path)
            ax.imshow(img)
            ax.set_title(f'Class {i}')
            ax.axis('off')

plt.tight_layout()
plt.savefig('sample_signs.png')
print("Sample signs saved as 'sample_signs.png'")
plt.close()

# ============================================================
# SECTION 3: ANALYZE IMAGE QUALITY
# ============================================================

print("\n" + "=" * 50)
print("ANALYZING IMAGE QUALITY")
print("=" * 50)

widths = train_df['Width'].values
heights = train_df['Height'].values

print(f"\nImage size statistics:")
print(f"Minimum width: {widths.min()} pixels")
print(f"Maximum width: {widths.max()} pixels")
print(f"Average width: {widths.mean():.1f} pixels")
print(f"\nMinimum height: {heights.min()} pixels")
print(f"Maximum height: {heights.max()} pixels")
print(f"Average height: {heights.mean():.1f} pixels")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(widths, bins=50, edgecolor='black')
plt.xlabel('Width (pixels)')
plt.ylabel('Number of Images')
plt.title('Distribution of Image Widths')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(heights, bins=50, edgecolor='black')
plt.xlabel('Height (pixels)')
plt.ylabel('Number of Images')
plt.title('Distribution of Image Heights')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('image_dimensions.png')
print("\nImage dimension analysis saved as 'image_dimensions.png'")
plt.close()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Comparison: Small vs Large Traffic Sign Images', fontsize=16)

small_images = train_df.nsmallest(5, 'Width')
for i, (idx, row) in enumerate(small_images.iterrows()):
    img_path = './gtsrb_data/' + row['Path']
    img = Image.open(img_path)
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Small: {row['Width']}x{row['Height']}")
    axes[0, i].axis('off')

large_images = train_df.nlargest(5, 'Width')
for i, (idx, row) in enumerate(large_images.iterrows()):
    img_path = './gtsrb_data/' + row['Path']
    img = Image.open(img_path)
    axes[1, i].imshow(img)
    axes[1, i].set_title(f"Large: {row['Width']}x{row['Height']}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('size_comparison.png')
print("Size comparison saved as 'size_comparison.png'")
plt.close()

print("\n" + "=" * 50)
print("QUALITY ANALYSIS COMPLETE")
print("=" * 50)
print("\nAll visualizations have been saved as PNG files in your project folder.")