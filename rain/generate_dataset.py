import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from random import gauss
import time
from synthetic_weather.rain.render_rain import render_rain

# Parameter configuration
mode = 'test'
startnum = 0 if mode == 'test' else 301
endnum = 600 if mode == 'test' else 350

datafolder = 'HeavyRainSynthetic'

os.makedirs(f'{mode}/in/', exist_ok=True)
os.makedirs(f'{mode}/streak/', exist_ok=True)
os.makedirs(f'{mode}/gt/', exist_ok=True)
os.makedirs('filelists/', exist_ok=True)

f_in = open(f'filelists/{mode}_in.txt', 'w')
f_st = open(f'filelists/{mode}_streak.txt', 'w')

# Set root directory
root_dir = r''
image_dir = os.path.join(root_dir, 'hld')

image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]


# Counter
counter = 1

for i in range(startnum, endnum + 1):
    fileindex = i
    imname = image_files[i]
    
    img = cv2.imread(os.path.join(image_dir, imname)).astype(np.float32) / 255.0

    cv2.imwrite(f'{mode}/gt/im_{i:04d}.png', img * 255)

    seed = max(0, min(0.5, abs(gauss(0, 0.5))))
    # Five intensity levels
    density_intensity = [(-8, 1, 60), (-7, 1, 60), (-6, 1, 50), (-5, 1, 40), (-4, 1, 30)]
    for level, (density, intensity, l) in enumerate(density_intensity):
        theta = np.random.randint(70, 111)
        start_time = time.time()
        # Render raindrop effect using Gaussian blur
        im = gaussian_filter(img, sigma=seed)
        rain, streak = render_rain(im, theta, density, intensity, l)

        rain = (rain * 255).astype(np.uint8)
        streak = (streak * 255).astype(np.uint8)

        file_name_format = f'im_{fileindex:04d}_level{level + 1}_s{theta:02d}_d{density}_i{intensity}.png'
        cv2.imwrite(os.path.join(mode, 'in', file_name_format), rain)
        cv2.imwrite(os.path.join(mode, 'streak', file_name_format), streak)
        # Record file path
        f_in.write(f'../../../data/{datafolder}/{mode}/in/{file_name_format}\n')
        f_st.write(f'../../../data/{datafolder}/{mode}/streak/{file_name_format}\n')

        elapsed_time = time.time() - start_time
        print(f'Num: {counter}, time elapsed: {elapsed_time:.2f}, sigma: {seed}')

        counter += 1

f_in.close()
f_st.close()

