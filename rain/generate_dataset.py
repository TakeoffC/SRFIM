# 只有雨的加成
import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from random import gauss
import time
from datagen.render_haze import render_haze
from datagen.render_rain import render_rain

# 参数配置
mode = 'test3'  # 或者 'val'
startnum = 0 if mode == 'test3' else 301
endnum = 600 if mode == 'test3' else 350

# 图片尺寸
datafolder = 'HeavyRainSynthetic'
# x, y, dx, dy = 0, 0, 720, 480

# 判断路径是否存在
# 如果不存在，创建相应的目录
os.makedirs(f'{mode}/in/', exist_ok=True)
os.makedirs(f'{mode}/streak/', exist_ok=True)
os.makedirs(f'{mode}/gt/', exist_ok=True)
os.makedirs('filelists/', exist_ok=True)

# 打开文件列表, 不存在则创建
f_in = open(f'filelists/{mode}_in.txt', 'w')
f_st = open(f'filelists/{mode}_streak.txt', 'w')

# 设置目录
root_dir = r'D:\Repetition\HeavyRainRemoval-master'
image_dir = os.path.join(root_dir, 'hld_1')

# 获取图像文件列表
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]


# 计数器
counter = 1

# 遍历每一张图像
for i in range(startnum, endnum + 1):
    fileindex = i
    imname = image_files[i]

    # 读取图像和深度图
    img = cv2.imread(os.path.join(image_dir, imname)).astype(np.float32) / 255.0

    # 保存真实图像（ground truth）
    # 保存文件和文件名, im_0001.png
    cv2.imwrite(f'{mode}/gt/im_{i:04d}.png', img * 255)

    # 随机数, 范围(0-1)
    seed = max(0, min(0.5, abs(gauss(0, 0.5))))
    # 五个强度
    density_intensity = [(-8, 1, 60), (-7, 1, 60), (-6, 1, 50), (-5, 1, 40), (-4, 1, 30)]
    for level, (density, intensity, l) in enumerate(density_intensity):
        theta = np.random.randint(70, 111)
        start_time = time.time()
        # 渲染雨滴效果
        # 使用高斯模糊
        im = gaussian_filter(img, sigma=seed)
        rain, streak = render_rain(im, theta, density, intensity, l)

        rain = (rain * 255).astype(np.uint8)
        streak = (streak * 255).astype(np.uint8)

        # 创建文件名格式化字符串4
        file_name_format = f'im_{fileindex:04d}_level{level + 1}_s{theta:02d}_d{density}_i{intensity}.png'
        # 保存图像
        cv2.imwrite(os.path.join(mode, 'in', file_name_format), rain)
        cv2.imwrite(os.path.join(mode, 'streak', file_name_format), streak)
        # 记录文件路径
        f_in.write(f'../../../data/{datafolder}/{mode}/in/{file_name_format}\n')
        f_st.write(f'../../../data/{datafolder}/{mode}/streak/{file_name_format}\n')

        # 打印信息
        elapsed_time = time.time() - start_time
        print(f'Num: {counter}, time elapsed: {elapsed_time:.2f}, sigma: {seed}')

        # 更新计数器
        counter += 1

# 关闭文件列表
f_in.close()
f_st.close()

