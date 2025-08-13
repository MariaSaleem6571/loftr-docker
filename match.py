import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from kornia_moons.viz import draw_LAF_matches
import uuid

def load_image_cv(path):
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

    if len(mkpts0) >= 4:
        H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
    else:
        H = None

    if H is not None:
        H4 = np.eye(4)
        H4[:3, :3] = H
        H4[:3, 3] = [0, 0, 0] 
    else:
        H4 = np.eye(4)  

    pair_id = f"{image_files[i].replace('.png','')}_{image_files[i+1].replace('.png','')}"

    row = {
        "uuid": str(uuid.uuid4()),
        "image1_index": i + 1,
        "image2_index": i + 2,
        "pair_id": pair_id,
        "r11": H4[0, 0], "r12": H4[0, 1], "r13": H4[0, 2], "tx": H4[0, 3],
        "r21": H4[1, 0], "r22": H4[1, 1], "r23": H4[1, 2], "ty": H4[1, 3],
        "r31": H4[2, 0], "r32": H4[2, 1], "r33": H4[2, 2], "tz": H4[2, 3],
        "h41": H4[3, 0], "h42": H4[3, 1], "h43": H4[3, 2], "h44": H4[3, 3],
    }
    results.append(row)

csv_path = os.path.join(output_dir, "homographies_loftr_batch2.csv")
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
print(f"Homography CSV saved at: {csv_path}")
