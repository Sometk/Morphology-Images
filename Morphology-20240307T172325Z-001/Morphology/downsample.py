import os
import numpy as np
import tinybrain
from PIL import Image

def downsample_image(input_path, output_path):
    img = Image.open(input_path)
    original_width, original_height = img.size  # Store original dimensions
    img_array = np.array(img)

    downsampled_array = tinybrain.downsample_with_averaging(img_array, factor=(2,2), num_mips=1, sparse=False)

    downsampled_img = Image.fromarray(downsampled_array[0])
    downsampled_img.save(output_path)

    # Check if dimensions have changed after downsampling
    downsampled_width, downsampled_height = downsampled_img.size
    if original_width != downsampled_width or original_height != downsampled_height:
        print(f"Image {input_path} has been successfully downsampled.")
    else:
        print(f"Image {input_path} has not been downsampled.")

input_folder = r"C:\Users\Banan\Morphology-Images\Morphology-20240307T172325Z-001\Morphology\SHL18_MIP3"
output_folder = r"C:\Users\Banan\Morphology-Images\Morphology-20240307T172325Z-001\Morphology\SHL18_DownSamp"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        downsample_image(input_path, output_path)
