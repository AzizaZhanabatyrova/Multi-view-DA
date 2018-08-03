import os
import os.path as osp
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import math

class sourceDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), random_rotate=True, random_flip=True, random_lighting=True, random_scaling=True, random_blur=True, scale = 1, ignore_label=0):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.random_rotate = random_rotate
        self.random_lighting = random_lighting
        self.random_scaling = random_scaling

        if random_lighting:
            crop_height, crop_width = self.crop_size
            self.gaussian = np.random.random((crop_height, crop_width, 1)).astype(np.float32)
            self.gaussian = np.concatenate((self.gaussian, self.gaussian, self.gaussian), axis = 2)

        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.random_flip = random_flip
        self.random_blur = random_blur
        self.img_ids = [i_id for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        

        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

        for name in self.img_ids:
            if '\n' in name:
                name = name.replace('\n','')

            img_name = name.split(' ')[0]
            label_name = name.split(' ')[1]

            img_file = osp.join(self.root, "%s" % img_name)
            label_file = osp.join(self.root, "%s" % label_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        width = image.width
        height = image.height

        # RANDOM SCALING WITH RANGE
        if self.random_scaling:
            scaling = random.randint(0, 1)
            self.scale = random.uniform(1, 1.5)

            if scaling:
                image = image.resize((int(math.ceil(width*self.scale)), int(math.ceil(height*self.scale))), Image.BILINEAR)
                label = label.resize((int(math.ceil(width*self.scale)), int(math.ceil(height*self.scale))), Image.NEAREST)

        # RANDOM CROPPING
        width = image.width
        height = image.height
        crop_height, crop_width = self.crop_size
        
        start_height = random.randint(0, height - crop_height)
        start_width = random.randint(0, width - crop_width)
        end_height = start_height + crop_height
        end_width = start_width + crop_width

        image_cropped = image.crop(box = (start_width, start_height, end_width, end_height))
        label_cropped = label.crop(box = (start_width, start_height, end_width, end_height))

        # RANDOM ROTATION
        if self.random_rotate:
            rotation = random.randint(0, 3)
            if rotation == 0:
                angle = Image.ROTATE_270
            elif rotation == 1:
                angle = Image.ROTATE_90
            elif rotation == 2:
                angle = Image.ROTATE_180

            if rotation == 0 or rotation == 1 or rotation == 2:
                image_cropped = image_cropped.transpose(angle)
                label_cropped = label_cropped.transpose(angle)

        # RANDOM FLIPPING
        if self.random_flip:
            flip_leftright = random.randint(0, 1)
            flip_updown = random.randint(0, 1)

            if flip_leftright:
                image_cropped = image_cropped.transpose(Image.FLIP_LEFT_RIGHT)
                label_cropped = label_cropped.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_updown:
                image_cropped = image_cropped.transpose(Image.FLIP_TOP_BOTTOM)
                label_cropped = label_cropped.transpose(Image.FLIP_TOP_BOTTOM)

        image_cropped = np.asarray(image_cropped, np.float32)
        label_cropped = np.asarray(label_cropped, np.float32)

        # RANDOM LIGHTING
        if self.random_lighting:
            change_lighting = random.randint(0, 1)

            if change_lighting:
                image_cropped = cv.addWeighted(image_cropped, 0.75, 0.25 * self.gaussian, 0.25, 0)
                image_cropped = image_cropped.astype("uint8")
                image_cropped = image_cropped.astype("float32")

        # RANDOM BLUR
        if self.random_blur:
            blur = random.randint(0, 1)

            if blur:
                image_cropped = cv.blur(image_cropped, (5,5))
                image_cropped = image_cropped.astype("uint8")
                image_cropped = image_cropped.astype("float32")


        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label_cropped.shape, dtype=np.float32)

        for k, v in self.id_to_trainid.items():
            label_copy[label_cropped == k] = v

        size = image_cropped.shape
        image_cropped = image_cropped[:, :, ::-1]  # change to BGR
        image_cropped -= self.mean
        image_cropped = image_cropped.transpose((2, 0, 1))

        return image_cropped.copy(), label_copy.copy(), np.array(size), name

if __name__ == '__main__':
    dst = sourceDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
