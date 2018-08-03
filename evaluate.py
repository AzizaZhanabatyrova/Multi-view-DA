import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_multi import Res_Deeplab
from dataset.val_dataset import valDataSet
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
from multiprocessing import Pool 
from utils.metric import ConfusionMatrix

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 0
CROP_SIZE = '550,550'
IMAGE_SIZE = '550,550'
NUM_LAYERS = 23
NUM_CLASSES = 5
NUM_STEPS = None
IMG_MEAN = np.array((84.96108839384782,  90.37637720260379, 93.43303655945203), dtype=np.float32) # ISPRS all BGR
palette = [[0, 0, 0], [82, 5, 9], [92, 32, 10], [90, 67, 17], [131, 34, 10]]
RESTORE_FROM = ""

def concatenate_side_by_side(list_images):
    return np.concatenate(list_images, axis = 2)

def colorize(pl, num_classes, pal, ignore_label):

    # pl stands for prediction or label
    # pallette in RGB

    assert num_classes == len(palette), "Number of colors in pallette does not correspond to number of classes"
    assert len(np.shape(pl)) == 2, "Prediction or label needs to have only two dimensions"

    dim1, dim2 = np.shape(pl)

    pl_col = np.zeros((3, dim1, dim2))
    #pl_col = np.concatenate([pl, pl, pl], axis = 0)

    for i in range(0, dim1):
        for j in range(0, dim2):
            cl = int(pl[i, j])
            if cl != ignore_label:
                pl_col[:, i, j] = pal[cl]

    return pl_col

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network - Evaluation")
    parser.add_argument("--crop-size", type=str, default=CROP_SIZE,
                        help="Size of the cropped images.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--image-size", type=str, default=IMAGE_SIZE,
                        help="Output image size.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS,
                        help="Number of layers in the model.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of steps.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

def get_iou(data_list, class_num, save_path=None):
    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    #print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list)+'\n')
            f.write(str(M)+'\n')
    return aveJ


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    h, w = map(int, args.crop_size.split(','))
    crop_size = (h, w)

    h, w = map(int, args.image_size.split(','))
    image_size = (h, w)

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    #model = Res_Deeplab(num_classes=args.num_classes, num_layers = args.num_layers, dropout = False)
    model = Res_Deeplab(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
        
    model.load_state_dict(saved_state_dict)

    model.train()
    model.cuda(gpu0)

    testloader = data.DataLoader(valDataSet(args.data_dir, 
                                           args.data_list, 
                                           max_iters=args.num_steps,
                                           crop_size=crop_size, 
                                           mean=IMG_MEAN, 
                                           scale=1, 
                                           mirror=False),
                                batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=image_size, mode='bilinear')

    data_list = []
    mIoU = 0
    for index, batch in enumerate(testloader):
        if index % 10 == 0:
            print('%d processed' % index)

        image, label, size, name = batch
        size = size[0].numpy()
        gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)

        _, output2 = model(Variable(image, volatile=True).cuda(gpu0))
        output = interp(output2).cpu().data[0].numpy()
        output = output[:,:size[0],:size[1]]

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        data_list.append([gt.flatten(), output.flatten()])

        '''output_col = colorize(output, args.num_classes, palette, args.ignore_label)
        label_col = colorize(np.squeeze(label, axis = 2), args.num_classes, palette, args.ignore_label)

        image = image.cpu().numpy()
        image = image[:,::-1,:,:]
        image = np.squeeze(image, axis = 0)
        image = image.transpose((1, 2, 0))
        image += IMG_MEAN
        image = image.transpose((2, 0, 1))

        name = name[0].split('/')[-1]
        #to_save = concatenate_side_by_side([image, label_col, output_col])
        #to_save.save('%s/%s_eval.png' % (args.save, name.split('.')[0]))'''
    mIoU = get_iou(data_list, args.num_classes)

    print("Final mIoU %f" % (mIoU))

if __name__ == '__main__':
    main()