import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import math

class valDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=1, mirror=True, ignore_label=0):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror

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

        # random cropping
        crop_height, crop_width = self.crop_size
        width = image.width
        height = image.height
        
        # FIXED SCALING
        image = image.resize((int(math.ceil(width*self.scale)), int(math.ceil(height*self.scale))), Image.BILINEAR)
        label = label.resize((int(math.ceil(width*self.scale)), int(math.ceil(height*self.scale))), Image.NEAREST)
        width = image.width
        height = image.height

        start_height = int(math.floor(height/2)) - int(math.floor(crop_height/2))
        start_width = int(math.floor(width/2)) - int(math.floor(crop_width/2))
        end_height = int(math.floor(height/2)) + int(math.floor(crop_height/2))
        end_width = int(math.floor(width/2)) + int(math.floor(crop_width/2))

        image_cropped = image.crop(box = (start_width, start_height, end_width, end_height))
        label_cropped = label.crop(box = (start_width, start_height, end_width, end_height))

        image_cropped = np.asarray(image_cropped, np.float32)
        label_cropped = np.asarray(label_cropped, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label_cropped.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label_cropped == k] = v

        size = image_cropped.shape
        image_cropped = image_cropped[:, :, ::-1]  # change to BGR
        image_cropped = image_cropped.transpose((2, 0, 1))

        return image_cropped.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = valDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
