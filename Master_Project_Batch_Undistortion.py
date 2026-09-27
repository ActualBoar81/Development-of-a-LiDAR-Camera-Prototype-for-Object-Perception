import cv2
import numpy as np
import os

# Load NEW calibration parameters
cameraMatrix = np.load("camera_matrix.npy")
distCoeffs = np.load("dist_coeffs.npy")

# Folders
input_folder = "calib_images"
output_folder = "undistorted_images"

os.makedirs(output_folder, exist_ok=True)

# Process images
for file in os.listdir(input_folder):

    if not file.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path)

    if img is None:
        print("Skipping:", file)
        continue

    h, w = img.shape[:2]

    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix,
        distCoeffs,
        (w, h),
        1,
        (w, h)
    )

    undistorted = cv2.undistort(
        img,
        cameraMatrix,
        distCoeffs,
        None,
        newCameraMatrix
    )

    output_path = os.path.join(
        output_folder,
        file
    )

    cv2.imwrite(output_path, undistorted)

print("Batch undistortion complete.")
