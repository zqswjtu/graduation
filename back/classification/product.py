# -*- coding: UTF-8 -*-
import csv
import glob
import os
import random
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Product(Dataset):
    def __init__(self, root, resize, mode="train"):
        super(Product, self).__init__()

        self.root = root
        self.resize = resize
        self.mode = mode

        self.label_map = {}  # "name": id
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.label_map[name] = len(self.label_map.keys())

        # print(self.label_map)
        # image, label
        self.images, self.labels = self.load_csv("images.csv")

        if mode == "train":
            # self.images, self.labels = self.images, self.labels
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == "val":
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        # else:
        #     self.images = self.images[int(0.8 * len(self.images)):]
        #     self.labels = self.labels[int(0.8 * len(self.labels)):]

        self.length = len(self.images)

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.label_map.keys():
                # "product\\shoe\\train_00005.jpg"
                images += glob.glob(os.path.join(self.root, name, "*.jpg"))
            print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode="w", newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.label_map[name]
                    # "product\\shoe\\train_00005.jpg", 3
                    writer.writerow([img, label])
                print("image path has been write into csv file:", filename)

        # read image path information from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return self.length

    def get_label_map(self):
        return self.label_map

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hat = (x - mean) / std
        # x = x_hat * std = mean
        # x.shape: [c, h, w]
        # mean.shape, std.shape: [3] ==> [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        out_x = x_hat * std + mean
        return out_x

    def __getitem__(self, idx):
        assert 0 <= idx < self.length
        # self.images, self.labels
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda p: Image.open(p).convert("RGB"),
            transforms.Resize((int(1.25 * self.resize), int(1.25 * self.resize))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            # 统计自imagenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

        img = tf(img)
        label = torch.tensor(label)
        return img, label
