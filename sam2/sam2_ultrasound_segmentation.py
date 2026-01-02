import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import instantiate
from skimage.filters import median
from skimage.morphology import disk
from skimage import exposure
from skimage.transform import rotate

# SAM2 imports
from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Paths ---
config_path = r"C:\Users\lenovo\Documents\SAM2\sam2\sam2\configs\sam2.1\sam2.1_hiera_s.yaml"
ckpt_path   = r"C:\Users\lenovo\Documents\SAM2\sam2\checkpoints\sam2.1_hiera_small.pt"
images_dir  = r"C:\Users\lenovo\Documents\SAM2\Processed\images"
masks_dir   = r"C:\Users\lenovo\Documents\SAM2\Processed\masks"
out_dir     = r"C:\Users\lenovo\Documents\SAM2\segmentation"
os.makedirs(out_dir, exist_ok=True)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Build SAM2 model ---
def build_sam2_yaml(config_file, ckpt_path=None, device="cuda", mode="eval"):
    cfg = OmegaConf.load(config_file)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

print("[INFO] Loading SAM2 model...")
model = build_sam2_yaml(config_path, ckpt_path, device)
predictor = SAM2ImagePredictor(model)
print("[INFO] SAM2 loaded successfully.")

# --- Metrics ---
def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return (2. * intersection) / (pred.sum() + gt.sum() + 1e-8)

def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)

# --- Preprocessing ---
def preprocess_image(gray_img):
    clahe_img = exposure.equalize_adapthist(gray_img)
    filtered  = median(clahe_img, disk(3))
    rgb       = np.repeat((filtered * 255).astype(np.uint8)[..., None], 3, axis=2)
    return rgb

# --- Augmentation ---
def augment_image(image, mode="flip"):
    if mode == "flip":
        return np.ascontiguousarray(np.fliplr(image))
    elif mode == "rotate":
        return np.ascontiguousarray(rotate(image, angle=15, resize=False))
    return np.ascontiguousarray(image)

# --- Process **all** images ---
image_files = sorted(
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
)
results = []

for img_name in image_files:
    img_path = os.path.join(images_dir, img_name)
    filename_base = os.path.splitext(img_name)[0]
    mask_name = f"mask_{filename_base.split('_')[-1]}.png"
    mask_path = os.path.join(masks_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"[WARN] GT mask missing for {img_name}, skipping...")
        continue

    # Load image + GT mask
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask_bin = (gt_mask > 127).astype(np.uint8)

    # Preprocess
    rgb = preprocess_image(gray)
    h, w = rgb.shape[:2]

    # --- Prepare prompts ---
    y, x = np.where(gt_mask_bin)
    center_point = np.array([[w // 2, h // 2]])
    bbox = np.array([[x.min(), y.min(), x.max(), y.max()]])
    fg_points = np.array([[x.min()+1, y.min()+1], [x.max()-1, y.max()-1]])
    bg_points = np.array([[0,0], [w-1,0], [0,h-1]])

    prompt_dict = {
        "Center-point": (center_point, np.array([1]), None),
        "Bounding-box": (None, None, bbox),
        "Foreground+Background": (np.vstack([fg_points, bg_points]),
                                  np.array([1,1,0,0,0]), None)
    }

    # Optional augmented evaluation
    rgb_aug = augment_image(rgb, mode="flip")
    prompt_dict["Augmented(BBox)"] = (None, None, bbox)

    # --- Predictions + metrics ---
    masks_results = {}
    for name, (points, labels, box) in prompt_dict.items():
        img_to_set = rgb if "Augmented" not in name else rgb_aug
        predictor.set_image(np.ascontiguousarray(img_to_set))
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)].astype(np.uint8)
        dsc = dice_score(best_mask, gt_mask_bin)
        iou = iou_score(best_mask, gt_mask_bin)
        masks_results[name] = (best_mask, dsc, iou)

    results.append((img_name, masks_results))

    # --- Save visualization ---
    fig, axes = plt.subplots(1, len(masks_results)+2, figsize=(16,4))
    axes[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image"); axes[0].axis("off")
    axes[1].imshow(gt_mask_bin, cmap="gray")
    axes[1].set_title("Ground Truth"); axes[1].axis("off")
    for i, (name, (mask, dsc, iou)) in enumerate(masks_results.items()):
        axes[i+2].imshow(mask, cmap="gray")
        axes[i+2].set_title(f"{name}\nDSC={dsc:.3f}, IoU={iou:.3f}")
        axes[i+2].axis("off")

    save_path = os.path.join(out_dir, f"{filename_base}_result.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

# --- Summary ---
print("\n=== Zero-Shot SAM2 Results (All Images) ===")
for img_name, mask_dict in results:
    print(f"\nImage: {img_name}")
    for name, (_, dsc, iou) in mask_dict.items():
        print(f"{name:25s}: Dice={dsc:.3f}, IoU={iou:.3f}")
