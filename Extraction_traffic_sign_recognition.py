import zipfile
import os

# Path to your downloaded zip file
zip_path = r"C:\Users\Collins\Downloads\traffic_sign_dataset.zip"

# Where we'll extract the data (in the same folder as your Python file)
extract_path = "./gtsrb_data"

# Create extraction directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract the zip file
print("Extracting dataset... This may take a minute.")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("✓ Extraction complete!")

# Let's explore what's inside
print("\n" + "=" * 50)
print("FOLDER STRUCTURE:")
print("=" * 50)

for root, dirs, files in os.walk(extract_path):
    level = root.replace(extract_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent} {os.path.basename(root)}/')

    # Only show first 2 levels to avoid too much output
    if level < 2:
        subindent = ' ' * 2 * (level + 1)
        # Show first 5 files as examples
        for file in files[:5]:
            print(f'{subindent} {file}')
        if len(files) > 5:
            print(f'{subindent}   ... and {len(files) - 5} more files')
