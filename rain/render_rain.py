import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.util import random_noise
import math


# Generate the convolution kernel and anchor point for motion blur
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
    # Blur kernel size
    sx = max(int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS)), 1)
    sy = max(int(math.fabs(length * sinalpha + psfwdt - length * EPS)), 1)
    psf1 = np.zeros((sy, sx))
    half = length / 2
    
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

    anchor = (0, 0)

    if 90 > angle > 0:
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, 0)
    elif -90 < angle < 0:  
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)
    elif angle < -90:  
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)
    psf1 = psf1 / psf1.sum()
    return psf1, anchor


def render_rain(img, theta, density, intensity, l):
    """
    Render rain effect on the image.

    Parameters:
    img -- Input image, Gaussian blurred image, numpy array format.
    theta -- Angle of inclination of the raindrop.
    density -- Density of the raindrops.
    intensity -- Intensity of the raindrops.

    Returns:
    image_rain -- Image with raindrop effects.
    actual_streak -- Actual image of the rain streaks.
    """
    image_rain = img.copy()
    h, w = img.shape[:2]

    # Generate random raindrop parameters
    s = 1.01 + np.random.rand() * 0.2  # Scaling factor to control the size of raindrops.
    m = density * (0.2 + np.random.rand() * 0.05)  # Mean of Gaussian noise to control the density of rain.
    v = intensity + np.random.rand() * 0.3  # Variance of Gaussian noise to control the intensity of the raindrops.

    # Generate noise seeds as raindrops
    dense_chnl_noise = random_noise(np.zeros((h, w, 1)), mode='gaussian', mean=m, var=v)

    # Adjust noise size to simulate raindrop size, scale the image
    dense_chnl_noise = cv2.resize(dense_chnl_noise, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    # Crop noise image to fit the original image size
    posv = np.random.randint(dense_chnl_noise.shape[0] - h)  # Ensure not to exceed the index in the vertical direction
    posh = np.random.randint(dense_chnl_noise.shape[1] - w)  # Ensure not to exceed the index in the horizontal direction
    # Ensure the cropped noise area matches the original image size
    dense_chnl_noise = dense_chnl_noise[posv:posv + h, posh:posh + w]


    # Create motion blur filter
    kernel, anchor = genaratePsf(l, theta)
    print(anchor)
    dense_chnl_motion = cv2.filter2D(dense_chnl_noise, -1, kernel, anchor=anchor)



    # Apply motion blur to create rain streaks
    # Filter negative values
    dense_chnl_motion[dense_chnl_motion < 0] = 0

    # Copy matrix to three channels
    dense_streak = np.repeat(dense_chnl_motion[:, :, np.newaxis], 3, axis=2)

    # Render image with rain streaks
    # Generate transparency coefficient
    tr = np.random.rand() * 0.05 + 0.04 * l + 0.2
    # Overlay raindrops onto the image
    image_rain = image_rain + tr * dense_streak
    # Limit pixel values not to exceed 1
    image_rain[image_rain >= 1] = 1

    # Calculate the actual raindrop image
    actual_streak = image_rain - img

    return image_rain, actual_streak
