# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import os.path


# 椒盐噪声
def salt_and_pepper_noise(src, percentage):
    sp_noise_img = src.copy()
    sp_noise_img = int(percentage * src.shape[0] * src.shape[1])
    for i in range(sp_noise_img):
        rand_r = np.random.randint(0, src.shape[0] - 1)
        rand_g = np.random.randint(0, src.shape[1] - 1)
        rand_b = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            sp_noise_img[rand_r, rand_g, rand_b] = 0
        else:
            sp_noise_img[rand_r, rand_g, rand_b] = 255
    return sp_noise_img


# 高斯噪声
def gaussian_noise(image, percentage):
    g_noise_img = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    g_noise_num = int(percentage * image.shape[0] * image.shape[1])
    for i in range(g_noise_num):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        g_noise_img[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return g_noise_img


# 昏暗
def darker(image, percentage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percentage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percentage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percentage)
    return image_copy


# 亮度
def brighter(image, percentage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percentage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percentage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percentage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image


if __name__ == "__main__":
    # 图片文件夹路径
    file_dir = r"../dataset/train/Toothbrush/"
    for img_name in os.listdir(file_dir):
        img_path = file_dir + img_name
        img = cv2.imread(img_path)
        # cv2.imshow("1",img)
        # cv2.waitKey(5000)
        # 旋转
        rotated_90 = rotate(img, 90)
        cv2.imwrite(file_dir + img_name[0:-4] + '_r90.jpg', rotated_90)
        rotated_180 = rotate(img, 180)
        cv2.imwrite(file_dir + img_name[0:-4] + '_r180.jpg', rotated_180)

    for img_name in os.listdir(file_dir):
        img_path = file_dir + img_name
        print(img_path)
        img = cv2.imread(img_path)
        # 镜像
        flipped_img = flip(img)
        cv2.imwrite(file_dir + img_name[0:-4] + '_fli.jpg', flipped_img)

        # 增加噪声
        # img_salt = salt_and_pepper_noise(img, 0.3)
        # cv2.imwrite(file_dir + img_name[0:7] + '_salt.jpg', img_salt)
        img_gauss = gaussian_noise(img, 0.1)
        cv2.imwrite(file_dir + img_name[0:-4] + '_noise.jpg', img_gauss)

        # 变亮、变暗
        img_darker = darker(img)
        cv2.imwrite(file_dir + img_name[0:-4] + '_darker.jpg', img_darker)
        img_brighter = brighter(img)
        cv2.imwrite(file_dir + img_name[0:-4] + '_brighter.jpg', img_brighter)

        blur = cv2.GaussianBlur(img, (7, 7), 1.5)
        #      cv2.GaussianBlur(图像，卷积核，标准差）
        cv2.imwrite(file_dir + img_name[0:-4] + '_blur.jpg', blur)
