import os
from PIL import Image
import numpy as np
import glob

dataset_path = r"C:\Users\lenovo\Documents\SAM2\Dataset"   # input dataset path
processed_path = r"C:\Users\lenovo\Documents\SAM2\Processed"
resize_shape = (512, 512)   # target size for SAM2

# Create output dirs
img_out = os.path.join(processed_path, "images")
mask_out = os.path.join(processed_path, "masks")
os.makedirs(img_out, exist_ok=True)
os.makedirs(mask_out, exist_ok=True)

# Collect images and masks
all_pngs = sorted(glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True))

images = [f for f in all_pngs if "mask" not in f.lower()]
masks  = [f for f in all_pngs if "mask" in f.lower()]

print(f"Found {len(images)} images and {len(masks)} masks")

# Ensure same count
if len(images) != len(masks):
    print(" Warning: Images and masks count differ. Pairing will be order-based.")

# Process each pair
for idx, (img_path, mask_path) in enumerate(zip(sorted(images), sorted(masks))):
    # --- Process Image ---
    img = Image.open(img_path).convert("RGB")
    img = img.resize(resize_shape, Image.BILINEAR)
    img_arr = np.array(img).astype("float32") / 255.0  # normalize [0,1]

    # save normalized image back to PNG
    img_out_path = os.path.join(img_out, f"image_{idx:04d}.png")
    Image.fromarray((img_arr * 255).astype(np.uint8)).save(img_out_path)

    # --- Process Mask ---
    mask = Image.open(mask_path).convert("L")  # grayscale
    mask = mask.resize(resize_shape, Image.NEAREST)  # keep labels intact
    mask_arr = np.array(mask)
    mask_bin = (mask_arr > 127).astype(np.uint8) * 255  # binarize

    mask_out_path = os.path.join(mask_out, f"mask_{idx:04d}.png")
    Image.fromarray(mask_bin).save(mask_out_path)

print("Preprocessing done!")
print("Processed images saved to:", img_out)
print("Processed masks saved to:", mask_out)
