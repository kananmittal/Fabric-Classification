import os
import shutil
from pathlib import Path

# Path to the raw dataset
src_dir = Path("archive")
dst_dir = Path("dataset_sorted")

# Create output directory
dst_dir.mkdir(parents=True, exist_ok=True)

# Loop through all .jpg images in the archive folder
for img_path in src_dir.glob("*.jpg"):
    # Extract class name from filename prefix, e.g., class01, class02
    class_name = img_path.stem.split("_")[0]
    
    # Create subdirectory for this class
    class_folder = dst_dir / class_name
    class_folder.mkdir(exist_ok=True)
    
    # Copy image to its class folder
    shutil.copy(img_path, class_folder / img_path.name)

print("✅ Images have been sorted into folders by class.")
