import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import instantiate

from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------- Metrics ----------
def dice_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    return (2.0 * inter) / (gt.sum() + pred.sum() + 1e-8)

def iou_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return inter / (union + 1e-8)


# ---------- Build SAM2 ----------
def build_sam2_yaml(config_file, ckpt_path=None, device="cuda", mode="eval"):
    cfg = OmegaConf.load(config_file)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def run_experiment_matplotlib(predictor, images_dir, masks_dir, out_dir):
    """
    Save results in a single figure:
      [ Input | Ground Truth | Point Prediction | Box Prediction ]
    with titles on top including Dice/IoU for point and box.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- Process ALL images with common extensions ----
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    img_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]
    )

    for fname in img_files:
        img_path  = os.path.join(images_dir, fname)
        # mask name pattern: mask_XXXX.png (second token after underscore)
        mask_path = os.path.join(masks_dir, "mask_" + fname.split("_")[1])

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt_mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None or gt_mask is None:
            print(f"[WARN] Missing file for {fname}, skipping...")
            continue

        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        predictor.set_image(img_rgb)
        h, w = img_rgb.shape[:2]
        gt_bin = (gt_mask > 127).astype(np.uint8)

        # ----- Point Prompt -----
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=True
        )
        pred_point = masks[np.argmax(scores)]
        d_point, i_point = dice_score(gt_bin, pred_point), iou_score(gt_bin, pred_point)

        # ----- Box Prompt -----
        input_box = np.array([w // 4, h // 4, 3 * w // 4, 3 * h // 4])
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        pred_box = masks[np.argmax(scores)]
        d_box, i_box = dice_score(gt_bin, pred_box), iou_score(gt_bin, pred_box)

        print(f"{fname} | Point: Dice={d_point:.3f}, IoU={i_point:.3f} | "
              f"Box: Dice={d_box:.3f}, IoU={i_box:.3f}")

        # ----- Matplotlib Figure -----
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        titles = [
            "Input Image",
            "Ground Truth",
            f"Point\nDSC={d_point:.3f}, IoU={i_point:.3f}",
            f"Box\nDSC={d_box:.3f}, IoU={i_box:.3f}"
        ]
        images = [
            img_rgb,
            gt_bin * 255,
            pred_point.astype(np.uint8) * 255,
            pred_box.astype(np.uint8) * 255
        ]

        for ax, img, title in zip(axes, images, titles):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{fname}_point_box.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved figure: {save_path}")


if __name__ == "__main__":
    # ----- Paths -----
    config     = r"C:\Users\lenovo\Documents\SAM2\sam2\sam2\configs\sam2.1\sam2.1_hiera_s.yaml"
    ckpt       = r"C:\Users\lenovo\Documents\SAM2\sam2\checkpoints\sam2.1_hiera_small.pt"
    images_dir = r"C:\Users\lenovo\Documents\SAM2\Processed\images"
    masks_dir  = r"C:\Users\lenovo\Documents\SAM2\Processed\masks"
    out_dir    = r"C:\Users\lenovo\Documents\SAM2\experiment"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading SAM2 model once...")
    model = build_sam2_yaml(config, ckpt, device)
    predictor = SAM2ImagePredictor(model)
    print("[INFO] Model loaded successfully.")

    run_experiment_matplotlib(predictor, images_dir, masks_dir, out_dir)
