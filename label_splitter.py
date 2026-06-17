import cv2
import numpy as np
import os

# Load the aggregate mask exported from CVAT
image_path = "frame_0000.png"
mask = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Define the CVAT colors for your labels (Note: OpenCV loads as BGR, not RGB)
# You can find exact RGB values in the 'labelmap.txt' CVAT includes in the export zip
labels = {
    'cyan_lobe': [225, 228, 48],  # Example BGR for Cyan
    'pink_lobe': [203, 192, 255],  # Example BGR for Pink
    'green_lobe': [113, 204, 60],  # Example BGR for Green
    'purple_tool': [255, 178, 163]  # Example BGR for Purple
}

for label_name, bgr_color in labels.items():
    # Create a binary mask where only the target color is 255 (white)
    lower_bound = np.array(bgr_color)
    upper_bound = np.array(bgr_color)
    binary_mask = cv2.inRange(mask, lower_bound, upper_bound)

    # Save the individual label mask
    output_filename = f"{label_name}_{os.path.basename(image_path)}"
    cv2.imwrite(output_filename, binary_mask)
    print(f"Saved: {output_filename}")