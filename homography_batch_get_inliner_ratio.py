import numpy as np
import torch
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import kornia as K
import kornia.feature as KF

def load_image_cv(path):
    print("Loading:", path)
    img_cv = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor

image_dir = "images"
output_dir = "output_batch_4_v4"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))

matcher = KF.LoFTR(pretrained="outdoor")

results = []
target_size = (480, 360)  # (height, width)

for i in range(len(image_files) - 1):
    fname1 = os.path.join(image_dir, image_files[i])
    fname2 = os.path.join(image_dir, image_files[i + 1])

    img1 = load_image_cv(fname1)
    img2 = load_image_cv(fname2)

    img1 = K.geometry.resize(img1, target_size, antialias=True)
    img2 = K.geometry.resize(img2, target_size, antialias=True)

    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),
        "image1": K.color.rgb_to_grayscale(img2),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    conf = correspondences["confidence"].cpu().numpy()

    norm_conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    colors = cm.viridis(norm_conf)[:, :3]

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

    img1_np = (img1[0].permute(1,2,0).numpy()*255).astype(np.uint8)
    img2_np = (img2[0].permute(1,2,0).numpy()*255).astype(np.uint8)

    h1, w1, _ = img1_np.shape
    h2, w2, _ = img2_np.shape
    max_h = max(h1, h2)
    total_w = w1 + w2
    canvas = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    canvas[:h1, :w1, :] = img1_np
    canvas[:h2, w1:w1+w2, :] = img2_np

    fig, ax = plt.subplots(figsize=(total_w/100, max_h/100), dpi=100)
    ax.imshow(canvas)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_aspect('equal')  

    for pt0, pt1, c in zip(mkpts0, mkpts1, colors):
        x0, y0 = pt0
        x1, y1 = pt1
        x1 += w1  
        ax.plot([x0, x1], [y0, y1], color=c, linewidth=1.0)

    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=conf.min(), vmax=conf.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Match Confidence", fontsize=12)

    pair_id = f"{image_files[i].replace('.png','')}_{image_files[i+1].replace('.png','')}"
    out_img_path = os.path.join(output_dir, f"match_{pair_id}.png")
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    

    results.append({
        "pair_id": pair_id,
        "image_1": image_files[i],
        "image_2": image_files[i+1],
        "total_matches": total_matches,
        "inlier_matches": num_inliers,
        "inlier_ratio_percent": round(inlier_ratio * 100, 2)
    })

csv_path = os.path.join(output_dir, "loftr_match_results.csv")
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
print(f"Done! CSV saved at: {csv_path}")

