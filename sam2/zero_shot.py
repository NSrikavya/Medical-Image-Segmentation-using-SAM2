import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import instantiate

from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Paths ---
config    = r"C:\Users\lenovo\Documents\SAM2\sam2\sam2\configs\sam2.1\sam2.1_hiera_s.yaml"
ckpt      = r"C:\Users\lenovo\Documents\SAM2\sam2\checkpoints\sam2.1_hiera_small.pt"
images_dir= r"C:\Users\lenovo\Documents\SAM2\Processed\images"
masks_dir = r"C:\Users\lenovo\Documents\SAM2\Processed\masks"
out_dir   = r"C:\Users\lenovo\Documents\SAM2\zero-shot output"
os.makedirs(out_dir, exist_ok=True)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Build SAM2 ---
def build_sam2_yaml(config_file, ckpt_path=None, device="cuda", mode="eval"):
    cfg = OmegaConf.load(config_file)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

print("Loading SAM2 model...")
model = build_sam2_yaml(config, ckpt, device)
predictor = SAM2ImagePredictor(model)
print("Model loaded successfully.")

# --- Metrics ---
def dice_score(pred, gt):
    """Dice Similarity Coefficient."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)

def iou_score(pred, gt):
    """Intersection over Union."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)

# --- Process **all** images ---
image_files = sorted(
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
)
results = []

for img_name in image_files:
    img_path = os.path.join(images_dir, img_name)

    # Derive mask filename (image_0001.png → mask_0001.png)
    num       = img_name.split("_")[-1]       # e.g. "0001.png"
    mask_name = f"mask_{num}"                  # → "mask_0001.png"
    mask_path = os.path.join(masks_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"[WARN] GT mask missing for {img_name}, skipping...")
        continue

    # --- Load input + mask ---
    gray        = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    rgb         = np.repeat(gray[..., None], 3, axis=2)
    gt_mask     = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask_bin = (gt_mask > 127).astype(np.uint8)

    # --- SAM2 Prediction ---
    predictor.set_image(rgb)
    h, w = rgb.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    best = masks[np.argmax(scores)].astype(np.uint8)

    # --- Metrics ---
    dsc = dice_score(best, gt_mask_bin)
    iou = iou_score(best, gt_mask_bin)
    results.append((img_name, dsc, iou))

    # --- Save side-by-side plot ---
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Input Image"); ax[0].axis("off")

    ax[1].imshow(gt_mask_bin, cmap="gray")
    ax[1].set_title("Ground Truth"); ax[1].axis("off")

    ax[2].imshow(best, cmap="gray")
    ax[2].set_title(f"Prediction\nDSC={dsc:.3f}, IoU={iou:.3f}")
    ax[2].axis("off")

    save_path = os.path.join(out_dir, f"result_{num}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

# --- Print summary ---
print("\n=== Zero-Shot Results (All Images) ===")
for img_name, dsc, iou in results:
    print(f"{img_name}: Dice={dsc:.3f}, IoU={iou:.3f}")
