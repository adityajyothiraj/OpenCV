from __future__ import print_function
from __future__ import division

import cv2 as cv
import numpy as np
import os

# ===============================
# SET YOUR FOLDER PATH HERE
# ===============================
path = "HDR_Images"

# ===============================
# LOAD IMAGES + EXPOSURE TIMES
# ===============================
def loadExposureSeq(path):
    images = []
    times = []

    list_file = os.path.join(path, "list.txt")

    if not os.path.exists(list_file):
        print("ERROR: list.txt not found at:", list_file)
        exit()

    with open(list_file, "r") as f:
        content = f.readlines()

    base_shape = None

    for line in content:
        tokens = line.split()
        if len(tokens) != 2:
            continue

        img_name = tokens[0]
        exposure = float(tokens[1])

        img_path = os.path.join(path, img_name)
        img = cv.imread(img_path)

        if img is None:
            print("Skipping:", img_path)
            continue

        # set base size from first image
        if base_shape is None:
            base_shape = img.shape[:2]
        else:
            img = cv.resize(img, (base_shape[1], base_shape[0]))

        images.append(img)
        times.append(1.0 / exposure)

    return images, np.asarray(times, dtype=np.float32)


# ===============================
# LOAD DATA
# ===============================
print("Loading images from:", path)

images, times = loadExposureSeq(path)

if len(images) < 2:
    print("ERROR: Need at least 2 images for HDR")
    exit()

print("Loaded images:", len(images))

# ===============================
# HDR PROCESSING
# ===============================

# 1. Calibrate camera response
calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)

# 2. Merge HDR
merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)

# 3. Tone mapping (for display)
tonemap = cv.createTonemap(2.2)
ldr = tonemap.process(hdr)

# 4. Exposure fusion (alternative HDR method)
merge_mertens = cv.createMergeMertens()
fusion = merge_mertens.process(images)

# ===============================
# SAVE OUTPUTS
# ===============================

cv.imwrite("fusion.png", np.clip(fusion * 255, 0, 255).astype("uint8"))
cv.imwrite("ldr.png", np.clip(ldr * 255, 0, 255).astype("uint8"))
cv.imwrite("hdr.hdr", hdr)

print("HDR processing completed successfully!")
print("Saved files:")
print(" - fusion.png")
print(" - ldr.png")
print(" - hdr.hdr")