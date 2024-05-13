import numpy as np
import os
import cv2
import math

# Path for the original image and path for saving the images
img_path = r''
save_dir = r''

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

image = cv2.imread(img_path)
image = image / 255.0 

(row, col, chs) = image.shape

# Constants
A = 0.9
size = math.sqrt(max(row, col))


def AddHaz(img, center, size, beta, A):
    """Add fog effect to the image"""
    (row, col, chs) = img.shape
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img[j][l][:] = img[j][l][:] * td + A * (1 - td)
    return img


# Generate and save images with different fog effects
for i in range(5):
    beta = 0.02 * (i + 1)
    # Randomly select the center of fog
    center_x = np.random.uniform(row // 4, row // 1.5)
    center_y = np.random.uniform(col // 4, col // 1.5)
    center = (center_x, center_y)

    foggy_image = AddHaz(np.copy(image), center, size, beta, A)
    foggy_image = np.clip(foggy_image * 255, 0, 255) 
    foggy_image = foggy_image.astype(np.uint8)  
    img_name = os.path.join(save_dir, f"foggy_image_level_{i + 1}.jpg")
    cv2.imwrite(img_name, foggy_image)  
