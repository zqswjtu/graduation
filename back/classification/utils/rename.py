# -*- coding: UTF-8 -*-
import os

folder_path = '../dataset/train/Toothbrush/'

num = 0

if __name__ == '__main__':
    for file in os.listdir(folder_path):
        s = '%05d' % num  # 前面补零占位
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, str(s) + '.jpg'))
        num += 1
