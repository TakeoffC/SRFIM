import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.util import random_noise
import math


# 生成卷积核和锚点, 用于运动模糊
def genaratePsf(length, angle):
    EPS = np.finfo(float).eps
    alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1
    # 模糊核大小
    # sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
    # sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
    sx = max(int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS)), 1)
    sy = max(int(math.fabs(length * sinalpha + psfwdt - length * EPS)), 1)
    psf1 = np.zeros((sy, sx))
    half = length / 2
    # psf1是左上角的权值较大，越往右下角权值越小的核。
    # 这时运动像是从右下角到左上角移动
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
            rad = math.sqrt(i * i + j * j)
            if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp * temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    # 运动方向是往左上运动，锚点在（0，0）
    anchor = (0, 0)
    # 运动方向是往右上角移动，锚点一个在右上角    #同时，左右翻转核函数，使得越靠近锚点，权值越大
    if 90 > angle > 0:
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, 0)
    elif -90 < angle < 0:  # 同理：往右下角移动
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)
    elif angle < -90:  # 同理：往左下角移动
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)
    psf1 = psf1 / psf1.sum()
    return psf1, anchor


def render_rain(img, theta, density, intensity, l):
    """
    在图像上渲染雨效果。

    参数:
    img -- 输入图像，高斯模糊后的图像，numpy数组格式。
    theta -- 雨滴倾斜的角度。
    density -- 雨滴的密度。
    intensity -- 雨滴的强度。

    返回值:
    image_rain -- 带有雨滴效果的图像。
    actual_streak -- 实际的雨滴条纹图。
    """
    image_rain = img.copy()
    h, w = img.shape[:2]

    # 随机生成雨滴参数
    s = 1.01 + np.random.rand() * 0.2  # 缩放因子，用于控制雨滴的尺寸。s会在1.01到1.21之间变化。
    m = density * (0.2 + np.random.rand() * 0.05)  # 高斯噪声的平均值，用于控制雨的密度。density参数乘以一个[0.2, 0.25]范围内的随机数得到。
    v = intensity + np.random.rand() * 0.3  # 高斯噪声的方差，用于控制雨滴的强度。intensity加上一个[0, 0.3]范围内的随机数得到。
    # l = np.random.randint(20, 40)  # 动态模糊的长度, 用于控制雨滴的长度, [20, 60]范围内的随机整数。
    # l = 20

    # 生成噪声种子作为雨滴
    dense_chnl_noise = random_noise(np.zeros((h, w, 1)), mode='gaussian', mean=m, var=v)

    # 调整噪声大小来模拟雨滴的大小, 对图像进行缩放插值
    dense_chnl_noise = cv2.resize(dense_chnl_noise, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    # 裁剪噪声图像以适应原始图像大小
    posv = np.random.randint(dense_chnl_noise.shape[0] - h)  # 确保竖直方向不会超出索引
    posh = np.random.randint(dense_chnl_noise.shape[1] - w)  # 确保水平方向不会超出索引
    # 这里确保裁剪出的噪声区域的h和w与原图像一致
    dense_chnl_noise = dense_chnl_noise[posv:posv + h, posh:posh + w]


    # 创建运动模糊滤镜
    kernel, anchor = genaratePsf(l, theta)
    print(anchor)
    dense_chnl_motion = cv2.filter2D(dense_chnl_noise, -1, kernel, anchor=anchor)



    # 应用运动模糊以创建雨滴条纹
    # 过滤负值
    dense_chnl_motion[dense_chnl_motion < 0] = 0
    # print(dense_chnl_motion)
    # cv2.imshow('1', dense_chnl_motion)
    # # 等待，直到任意键被按下
    # cv2.waitKey(0)
    #
    # # 关闭所有OpenCV窗口
    # cv2.destroyAllWindows()
    # 复制矩阵到三个通道
    dense_streak = np.repeat(dense_chnl_motion[:, :, np.newaxis], 3, axis=2)

    # 渲染带有雨滴条纹的图像
    # 生成透明度系数
    tr = np.random.rand() * 0.05 + 0.04 * l + 0.2
    # 叠加雨滴到图像上
    image_rain = image_rain + tr * dense_streak
    # 限制像素值不超过1
    image_rain[image_rain >= 1] = 1

    # 计算实际的雨滴图像
    actual_streak = image_rain - img

    return image_rain, actual_streak


if __name__ == "__main__":
    im = cv2.imread(r'D:\Repetition\HeavyRainRemoval-master\test\0000.png').astype(np.float32) / 255.0
    rain, streak = render_rain(im, 100, -2, 2, 20)  # -4, 0.7
    cv2.imshow('motion', rain)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
