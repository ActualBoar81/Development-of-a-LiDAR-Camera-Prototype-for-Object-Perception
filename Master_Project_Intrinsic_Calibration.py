import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (8, 6)

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob("calib_images/*.jpg")

print("Images found:", len(images))

# Folder for visual verification
output_folder = "detected_corners"
os.makedirs(output_folder, exist_ok=True)

for fname in images:

    img = cv2.imread(fname)

    if img is None:
        print("Could not read:", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    print(os.path.basename(fname), ret)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw detected corners
        cv2.drawChessboardCorners(
            img,
            CHECKERBOARD,
            corners,
            ret
        )

        # Save verification image
        output_path = os.path.join(
            output_folder,
            os.path.basename(fname)
        )

        cv2.imwrite(output_path, img)

print("\nSuccessful detections:", len(objpoints))

if len(objpoints) == 0:
    print("No checkerboards detected!")
    exit()

# Use the image size from the last successfully read image
image_size = gray.shape[::-1]

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("\nCalibration RMS error:", ret)

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)

np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

print("\nSaved:")
print("camera_matrix.npy")
print("dist_coeffs.npy")

print("\nCalibration complete.")
