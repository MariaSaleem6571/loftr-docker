import cv2
import numpy as np
import os
import uuid
import pandas as pd

# Directory containing images
image_dir = "images"
output_csv = "homographies.csv"

# List all images in sorted order
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

MIN_MATCH_COUNT = 10
results = []

# Loop over consecutive image pairs
for i in range(len(image_files) - 1):
    img1_path = os.path.join(image_dir, image_files[i])
    img2_path = os.path.join(image_dir, image_files[i + 1])

    # Load grayscale images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Skipping {img1_path} or {img2_path} (cannot load)")
        continue

    # Detect SIFT features and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Convert 3×3 H to 4×4 matrix for CSV format
            H4 = np.eye(4)
            H4[:3, :3] = M[:3, :3]
            H4[:3, 3] = M[:3, 2]

            # Prepare row for CSV
            row = {
                "uuid": str(uuid.uuid4()),
                "image1_index": i + 1,
                "image2_index": i + 2,
                "pair_id": f"{os.path.splitext(image_files[i])[0]}_{os.path.splitext(image_files[i+1])[0]}",
                "r11": H4[0, 0], "r12": H4[0, 1], "r13": H4[0, 2], "tx": H4[0, 3],
                "r21": H4[1, 0], "r22": H4[1, 1], "r23": H4[1, 2], "ty": H4[1, 3],
                "r31": H4[2, 0], "r32": H4[2, 1], "r33": H4[2, 2], "tz": H4[2, 3],
                "h41": H4[3, 0], "h42": H4[3, 1], "h43": H4[3, 2], "h44": H4[3, 3],
            }
            results.append(row)
        else:
            print(f"⚠️ No homography found for {image_files[i]} and {image_files[i+1]}")
    else:
        print(f"⚠️ Not enough matches for {image_files[i]} and {image_files[i+1]}")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"✅ Homographies saved to {output_csv}")

