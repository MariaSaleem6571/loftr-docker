import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from kornia_moons.viz import draw_LAF_matches

fname1 = "images/image_0001.png"
fname2 = "images/image_0002.png"

# Load images using OpenCV and convert to RGB tensor normalized to [0,1]
def load_image_cv(path):
    print("Loading:", path)
    img_cv = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor

img1 = load_image_cv(fname1)
img2 = load_image_cv(fname2)

img1 = K.geometry.resize(img1, (600, 375), antialias=True)
img2 = K.geometry.resize(img2, (600, 375), antialias=True)

# LoFTR matcher (CPU-only)
matcher = KF.LoFTR(pretrained="outdoor")

# Convert to grayscale
input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),
    "image1": K.color.rgb_to_grayscale(img2),
}

# Run matcher
with torch.inference_mode():
    correspondences = matcher(input_dict)

# Extract keypoints
mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()

# Estimate fundamental matrix with MAGSAC++
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0

# Draw and save matches
draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={
        "inlier_color": (0.2, 1, 0.2),
        "tentative_color": None,
        "feature_color": (0.2, 0.5, 1),
        "vertical": False,
    },
)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "loftr_matches.png")
plt.savefig(output_path, dpi=300)
print(f"Saved output to {output_path}")

