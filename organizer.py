import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array

# Define paths to the subfolders
dataset_path = 'archive'  # Update this path if necessary
defect_folder = '/Users/rashmichawla/Downloads/archive/Defect_images'
no_defect_folder = '/Users/rashmichawla/Downloads/archive/NODefect_images'
mask_folder = '/Users/rashmichawla/Downloads/archive/Mask_images'


# Set image size for resizing
image_size = (128, 128)

def load_images_from_folder(folder, label, image_size):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = img_to_array(img)
            images.append(img)
            labels.append(label)
    return images, labels

# Load the images and labels for both defect and no-defect
defect_images, defect_labels = load_images_from_folder(defect_folder, 1, image_size)
no_defect_images, no_defect_labels = load_images_from_folder(no_defect_folder, 0, image_size)

# Combine the defect and no-defect images and labels
images = defect_images + no_defect_images
labels = defect_labels + no_defect_labels

# Convert to numpy arrays for ML
images = np.array(images, dtype="float32") / 255.0  # Normalize the images
labels = np.array(labels)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Save the processed dataset to a file
np.savez('dataset.npz', 
         X_train=X_train, y_train=y_train, 
         X_val=X_val, y_val=y_val)

print("✅ Dataset saved to dataset.npz")

