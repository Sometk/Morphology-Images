import os
import cv2
import numpy as np

def crop_image_only_outside(img, tol=0):
    mask = img > tol
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]

# Directory containing the image slices of the hydra
input_dir = "/Users/richardren/Morphology-Images/Morphology-20240307T172325Z-001/Morphology/NET10_MIP5_DS"
output_dir = "/Users/richardren/Morphology-Images/Morphology-20240307T172325Z-001/Morphology/NET10_MIP5_DSBS"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over the images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):  # Assuming images are PNG format
        img_path = os.path.join(input_dir, filename)
        
        # Read the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Binarize the image
        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Crop the binarized image
        cropped_img = crop_image_only_outside(binary_img, tol=0)
        
        # Save the processed image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped_img)
