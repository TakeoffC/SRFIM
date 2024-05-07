import numpy as np
import os
import cv2
import math

# 图片路径和保存目录
img_path = r'D:\Repetition\image-matching-toolbox-main\data\datasets\hld_1280_960_new2\i_architecture_1\1.jpg'
save_dir = r'D:\Repetition\image-matching-toolbox-main\test_fog'  # 设定保存目录

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取图像
image = cv2.imread(img_path)
image = image / 255.0  # 归一化到0-1范围

# 获取图像维度
(row, col, chs) = image.shape

# 常量设置
A = 0.9
size = math.sqrt(max(row, col))


def AddHaz(img, center, size, beta, A):
    """为图像添加雾效果"""
    (row, col, chs) = img.shape
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img[j][l][:] = img[j][l][:] * td + A * (1 - td)
    return img


# 生成不同雾效果的图像并保存
for i in range(5):
    beta = 0.02 * (i + 1)
    # 随机选择雾化中心
    center_x = np.random.uniform(row // 4, row // 1.5)
    center_y = np.random.uniform(col // 4, col // 1.5)
    center = (center_x, center_y)

    foggy_image = AddHaz(np.copy(image), center, size, beta, A)
    foggy_image = np.clip(foggy_image * 255, 0, 255)  # 重新缩放到0-255范围
    foggy_image = foggy_image.astype(np.uint8)  # 转换为整型
    img_name = os.path.join(save_dir, f"foggy_image_level_{i + 1}.jpg")
    cv2.imwrite(img_name, foggy_image)  # 保存图片
