import os
import shutil
import random

# Paths
base_dir = r"C:\Users\lenovo\Documents\SAM2\Processed"   
images_dir = os.path.join(base_dir, "images")
masks_dir = os.path.join(base_dir, "masks")

output_dir = r"C:\Users\lenovo\Documents\SAM2\split data"
splits = ["train", "val", "test"]

# Create output folders
for split in splits:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

# Collect all images (assume mask names match)
all_images = sorted(os.listdir(images_dir))

# Shuffle for randomness
random.seed(42)  # reproducibility
random.shuffle(all_images)

# Split indices: 70% train, 15% val, 15% test
n = len(all_images)
train_split = int(0.7 * n)
val_split = int(0.85 * n)

train_files = all_images[:train_split]
val_files = all_images[train_split:val_split]
test_files = all_images[val_split:]

splits_dict = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

# Copy files into respective folders
for split, file_list in splits_dict.items():
    for fname in file_list:
        # Image
        src_img = os.path.join(images_dir, fname)
        dst_img = os.path.join(output_dir, split, "images", fname)
        shutil.copy(src_img, dst_img)

        # Mask (same number suffix)
        mask_name = fname.replace("image_", "mask_")
        src_mask = os.path.join(masks_dir, mask_name)
        dst_mask = os.path.join(output_dir, split, "masks", mask_name)
        shutil.copy(src_mask, dst_mask)

print("Dataset successfully split into train/val/test!")
