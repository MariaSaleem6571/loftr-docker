import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from kornia_moons.viz import draw_LAF_matches

def load_image_cv(path):
    print("Loading:", path)
    img_cv = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor

image_dir = "images"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))

matcher = KF.LoFTR(pretrained="outdoor")

results = []

for i in range(len(image_files) - 1):
    fname1 = os.path.join(image_dir, image_files[i])
    fname2 = os.path.join(image_dir, image_files[i + 1])

    img1 = load_image_cv(fname1)
    img2 = load_image_cv(fname2)

    img1 = K.geometry.resize(img1, (600, 375), antialias=True)
    img2 = K.geometry.resize(img2, (600, 375), antialias=True)

    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),
        "image1": K.color.rgb_to_grayscale(img2),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()

    if len(mkpts0) >= 8:
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0
        num_inliers = inliers.sum()
        total_matches = len(inliers)
        inlier_ratio = num_inliers / total_matches
    else:
        inliers = np.zeros(len(mkpts0), dtype=bool)
        num_inliers = 0
        total_matches = len(mkpts0)
        inlier_ratio = 0.0

    print(f"{image_files[i]} ↔ {image_files[i + 1]} — Inliers: {num_inliers}/{total_matches} ({inlier_ratio:.2%})")

    fig = plt.figure()
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
    pair_id = f"{image_files[i].replace('.png','')}_{image_files[i+1].replace('.png','')}"
    out_img_path = os.path.join(output_dir, f"match_{pair_id}.png")
    plt.savefig(out_img_path, dpi=300)
    plt.close(fig)

    results.append({
        "pair_id": pair_id,
        "image_1": image_files[i],
        "image_2": image_files[i+1],
        "total_matches": total_matches,
        "inlier_matches": num_inliers,
        "inlier_ratio_percent": round(inlier_ratio * 100, 2)
    })

csv_path = os.path.join(output_dir, "inlier_results.csv")
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
print(f"\n All done. CSV saved at: {csv_path}")

