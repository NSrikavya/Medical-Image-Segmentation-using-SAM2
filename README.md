# Medical Image Segmentation using SAM2

Zero-shot segmentation of ultrasound images with Meta's **Segment Anything Model 2 (SAM 2.1)**. The project preprocesses an ultrasound dataset, prompts SAM2 with points and bounding boxes (no fine-tuning), and compares the predictions with ground-truth masks using **Dice (DSC)** and **IoU**.

## Overview

SAM2 is a promptable foundation model trained on natural images and video. This project tests how well it transfers to medical ultrasound images, where lesions are small, low-contrast and noisy. It evaluates:

- **Center-point prompt**: a single foreground click at the image center
- **Bounding-box prompt**: a box around the target region
- **Foreground + background points**: two foreground points and three background corner points
- **Augmented (flip) bounding box**: box prompt on a horizontally flipped image

Images are optionally preprocessed with CLAHE contrast enhancement and median filtering before being passed to the model.

## Repository Structure

```
Medical-Image-Segmentation-using-SAM2/
├── preprocessing.py                   # Resize to 512x512, binarize masks, rename pairs
├── split.py                           # 70/15/15 train/val/test split (seed 42)
├── segmentation/                      # Output figures from sam2_ultrasound_segmentation.py
└── sam2/                              # Meta's SAM 2 codebase + project scripts
    ├── zero_shot.py                   # Baseline: center-point prompt on raw images
    ├── experiment.py                  # Point vs. box prompt comparison
    ├── sam2_ultrasound_segmentation.py# Full pipeline: preprocessing + 4 prompt strategies
    ├── checkpoints/download_ckpts.sh  # Model weight download script
    └── sam2/configs/                  # SAM 2 / 2.1 model configs
```

The rest of `sam2/` (training, demo, notebooks, `sav_dataset`) is the unmodified upstream SAM 2 repository.

## Setup

**Requirements:** Python 3.10+, PyTorch 2.5.1+ and a CUDA GPU (recommended; CPU also works but is slow).

```bash
git clone https://github.com/<your-username>/Medical-Image-Segmentation-using-SAM2.git
cd Medical-Image-Segmentation-using-SAM2/sam2

pip install -e .
pip install opencv-python matplotlib scikit-image pillow numpy
```

Download the SAM 2.1 weights (the scripts use **Hiera-Small**):

```bash
cd checkpoints
./download_ckpts.sh        # or download sam2.1_hiera_small.pt manually
cd ..
```

## Dataset

The raw dataset is not included in the repository (`Dataset/` is git-ignored). Place your ultrasound images and masks in a single folder tree as `.png` files, where mask filenames contain the word `mask` (for example `img_01.png` and `img_01_mask.png`).

## Usage

### 1. Preprocess

Resizes images to 512x512 RGB, resizes masks with nearest-neighbour interpolation, binarizes them (threshold 127), and saves them as `image_XXXX.png` / `mask_XXXX.png`.

```bash
python preprocessing.py
```

> Image and mask pairing is **order-based** after sorting filenames. Make sure the sorted order of images and masks lines up.

### 2. (Optional) Split the data

```bash
python split.py
```

Creates `train/`, `val/` and `test/` folders with a 70/15/15 split and a fixed seed for reproducibility.

### 3. Run SAM2

```bash
cd sam2

# Baseline zero-shot with a center-point prompt
python zero_shot.py

# Point vs. box prompt comparison
python experiment.py

# Full pipeline with preprocessing and multiple prompt strategies
python sam2_ultrasound_segmentation.py
```

Each script saves side-by-side figures (input, ground truth, predictions with Dice and IoU in the titles) and prints per-image metrics to the console.

> **Paths:** the scripts currently use hard-coded Windows paths (`C:\Users\lenovo\Documents\SAM2\...`). Edit the path variables at the top of each script (and in `preprocessing.py` / `split.py`) to match your machine before running.

## Evaluation Metrics

| Metric | Formula |
|--------|---------|
| Dice (DSC) | 2·\|P ∩ G\| / (\|P\| + \|G\|) |
| IoU | \|P ∩ G\| / \|P ∪ G\| |

where P is the predicted mask and G is the ground-truth mask.

## Example Output

Each figure in `segmentation/` shows, left to right: the input image, ground truth, and the mask produced by each prompt strategy with its Dice/IoU. On a sample liver ultrasound image, a bounding-box prompt reached a Dice of about 0.89, while a single center-point prompt failed to localize the small lesion (Dice 0.00). This is typical of zero-shot SAM2 on small, low-contrast lesions: **box prompts are far more reliable than single points**.

## Notes and Limitations

- In `sam2_ultrasound_segmentation.py`, the bounding-box and foreground-point prompts are derived from the **ground-truth mask**, so those scores reflect an oracle-prompt setting and are an upper bound. A realistic deployment would need prompts from a user or a detector.
- Results are zero-shot; no fine-tuning of SAM2 is performed.
- This is a research and educational project and is **not** intended for clinical use.

## Acknowledgements

- [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2), Meta AI (FAIR), licensed under Apache 2.0
- Dataset: add the dataset name, source and license here

