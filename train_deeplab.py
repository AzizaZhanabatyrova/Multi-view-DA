
import argparse
import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data, model_zoo

import torchvision.utils as vutils
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from model.deeplab_multi3 import Res_Deeplab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.isprs_dataset import isprsDataSet
from dataset.source_dataset import sourceDataSet
from dataset.val_dataset import valDataSet
from PIL import Image

from multiprocessing import Pool 
from utils.metric import ConfusionMatrix

ADV_OPTION = 1
MODEL = 'DeepLab'
BATCH_SIZE = 1
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
DATA_DIRECTORY2 = './data/GTA5'
DATA_LIST_PATH2 = './dataset/gta5_list/train.txt'
DATA_DIRECTORY_TARGET = '/cluster/work/riner/users/zaziza/isprs'
DATA_LIST_PATH_TARGET = '/cluster/work/riner/users/zaziza/isprs/listsOfDataDirs/train.txt'
DATA_DIRECTORY_VAL = '/cluster/work/riner/users/zaziza/isprs'
DATA_LIST_PATH_VAL = '/cluster/work/riner/users/zaziza/isprs/listsOfDataDirs/val_greyscale.txt'
INPUT_SIZE = '550,550'
INPUT_SIZE_TARGET = '550,550'
IGNORE_LABEL = 0
ITER_SIZE = 1
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
MOMENTUM = 0.9
NUM_VAL_IMAGES = 50
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 80000  # early stopping
NUM_MODELS_KEEP = 60
NUM_WORKERS = 4
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/cluster/work/riner/users/zaziza/snapshots/AdaptSegNet/DeepLab_resnet_pretrained_init-f81d91e8.pth' # ImageNet
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1000
SAVE_PATH = './result/cityscapes'
SNAPSHOT_DIR = './snapshots/'
SOURCE = 'cityscapes'
TRAINING_OPTION = 1
VAL_EVERY = 10
WEIGHT_DECAY = 0.0005

palette = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 255], [255, 255, 0]]

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--adv-option", type=int, default=ADV_OPTION,
                        help="One of options for adversarial training of the generator")
    parser.add_argument("--augment_2", action="store_true",
                        help="Whether to augment source dataset")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--source", type=str, default=SOURCE,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir2", type=str, default=DATA_DIRECTORY2,
                        help="Path to the directory containing the source dataset 2.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data-list2", type=str, default=DATA_LIST_PATH2,
                        help="Path to the file listing the images in the source dataset 2.")
    parser.add_argument("--data-dir-val", type=str, default=DATA_DIRECTORY_VAL,
                        help="Path to the directory containing the tarez dataset.")
    parser.add_argument("--data-list-val", type=str, default=DATA_LIST_PATH_VAL,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--num-val-images", type=int, default=NUM_VAL_IMAGES,
                        help="Number of validation images.")
    parser.add_argument("--num-models-keep", type=int, default=NUM_MODELS_KEEP,
                        help="Number of validation images.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--training-option", type=int, default=TRAINING_OPTION,
                        help="1 or 2.")
    parser.add_argument("--val-every", type=int, default=VAL_EVERY,
                        help="Validate every # number of iterations.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()

if args.source == 'cityscapes':
    IMG_MEAN_SOURCE2 = np.array((72.3923987619416, 82.90891754262587, 73.15835921071157), dtype=np.float32) # cityscapes BGR
elif args.source == 'airsim':
    IMG_MEAN_SOURCE2 = np.array((175.4872285082577, 194.67600866800535, 189.40416391663257), dtype=np.float32) # airsim BGR
elif args.source == 'vistas':
    IMG_MEAN_SOURCE2 = np.array((119.83100494161111, 116.921607975, 106.85407974777779), dtype=np.float32) # vistas BGR

def colorize(pl, num_classes, pal, ignore_label):
    assert num_classes == len(palette), "Number of colors in pallette does not correspond to number of classes"
    assert len(np.shape(pl)) == 2, "Prediction or label needs to have only two dimensions"
    dim1, dim2 = np.shape(pl)
    pl_col = np.zeros((3, dim1, dim2))
    for i in range(0, dim1):
        for j in range(0, dim2):
            cl = int(pl[i, j])
            if cl != ignore_label:
                pl_col[:, i, j] = pal[cl]
    return pl_col

def loss_calc(pred, label, gpu, ignore_label, train_name):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    try:
        label = Variable(label.long()).cuda(gpu)
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda(gpu)
        return criterion(pred, label)
    except RuntimeError as e:
        print('e', e)
        print("RuntimeError", train_name)
        return 0

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)


    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def miou(pred, target, n_classes = 20, ignore_classes = [0]):

    ious = [] # IoUs

    pred = np.asarray(pred)
    target = np.asarray(target)
    pred = pred.flatten()
    target = target.flatten()

    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)

    for cls in range(0, n_classes):
        if cls not in ignore_classes:
            pred_inds = pred == cls # Find where in the predictions we have class = cls 
            target_inds = target == cls  # Find where in the target we have class = cls 
            intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]
            union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
            ious.append(float(intersection) / float(max(union, 1)))
    return np.mean(ious) # Return mean of all classes


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
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list)+'\n')
            f.write(str(M)+'\n')
    return aveJ

def validation(valloader, model, interp_target, writer, i_iter, chosen_indices):
    mIoU = 0
    output_col = []
    label_val_col = []
    image_val_chosen = []
    data_list = []
    for index, batch_val in enumerate(valloader):
        image_val, label_val, size, name_val = batch_val

        output = model(Variable(image_val, volatile=True).cuda(args.gpu))
        output = interp_target(output).cpu().data[0].numpy()
        size = size[0].numpy()
        output = output[:,:size[0],:size[1]]

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        if index in chosen_indices:
            output_col.append(colorize(output, args.num_classes, palette, args.ignore_label))
            label_val_col.append(colorize(np.squeeze(label_val, axis = 2), args.num_classes, palette, args.ignore_label))

        image_val = image_val.numpy()
        image_val = image_val[:,::-1,:,:]

        if index in chosen_indices:
            image_val_chosen_cp = np.copy(image_val)
            image_val_chosen.append(np.squeeze(image_val_chosen_cp, axis = 0))

        gt = np.asarray(label_val[0].numpy()[:size[0],:size[1]], dtype=np.int)

        data_list.append([gt.flatten(), output.flatten()])

    mIoU = get_iou(data_list, args.num_classes)

    # Save images for tensorboard
    image_collection = []
    for i in range(len(chosen_indices)):
        image_collection.append(concatenate_side_by_side([image_val_chosen[i], label_val_col[i], output_col[i]]))

    image_to_save = concatenate_above_and_below(image_collection)
    image_to_save = np.transpose(image_to_save, (1, 2, 0))
    image_to_save = vutils.make_grid(transforms.ToTensor()(image_to_save), normalize=True, scale_each=True)

    writer.add_scalar('miou', mIoU, i_iter)
    writer.add_image('Image/label/pred', image_to_save, i_iter)
    return mIoU

def concatenate_side_by_side(list_images):
    return np.concatenate(list_images, axis = 2)

def concatenate_above_and_below(list_images):
    return np.concatenate(list_images, axis = 1)

def main():
    """Create the model and start the training."""
    model_num = 0 # The number of model (for saving models)

    torch.manual_seed(args.random_seed)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    writer = SummaryWriter(log_dir = args.snapshot_dir)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu
    cudnn.benchmark = True

    # init G
    if args.model == 'DeepLab':

        model = Res_Deeplab(num_classes=args.num_classes)

        saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()

        for i in saved_state_dict:
            i_parts = i.split('.')
            if args.not_restore_last == True:
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            else:
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]        
        model.load_state_dict(new_params)

    model.train()
    model.cuda(args.gpu)

    trainloader2 = data.DataLoader(sourceDataSet(args.data_dir2, 
                                                    args.data_list2, 
                                                    max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                    crop_size=input_size,
                                                    random_rotate=False, 
                                                    random_flip=args.augment_2, 
                                                    random_lighting=args.augment_2,
                                                    random_blur=args.augment_2,
                                                    random_scaling=args.augment_2,
                                                    mean=IMG_MEAN_SOURCE2,
                                                    ignore_label=args.ignore_label,
                                                    source=args.source),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter2 = enumerate(trainloader2)

    valloader = data.DataLoader(valDataSet(args.data_dir_val, 
                                           args.data_list_val, 
                                           crop_size=input_size, 
                                           mean=IMG_MEAN_SOURCE2, 
                                           mirror=False,
                                           source=args.source),
                                batch_size=1, shuffle=False, pin_memory=True)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate }, 
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear')

    # List saving all best 5 mIoU's 
    best_mIoUs = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i_iter in range(args.num_steps):
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        while True:
            try:

                _, batch = next(trainloader_iter2) 
                images2, labels2, _, train_name2 = batch
                images2 = Variable(images2).cuda(args.gpu)

                pred2 = model(images2)
                pred2 = interp(pred2)
                print(pred2)
                print(labels2)

                print(pred2.size())
                print(labels2.size())

                loss = loss_calc(pred2, labels2, args.gpu, args.ignore_label, train_name2)
                loss.backward()

                break
            except (RuntimeError, AssertionError, AttributeError):
                continue
        print('Iter ...')
        optimizer.step()



        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            #print ('taking snapshot ...')
            if model_num != args.num_models_keep:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'model_' + str(model_num) + '.pth'))
                model_num = model_num +1
            if model_num == args.num_models_keep:
                model_num = 0

        # Validation
        if (i_iter % args.val_every== 0 and i_iter != 0) or i_iter == 1:
            mIoU = validation(valloader, model, interp, writer, i_iter, [37,41,10])
            for i in range(0, len(best_mIoUs)):
                if best_mIoUs[i] < mIoU:
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'bestmodel_' + str(i) + '.pth'))
                    best_mIoUs.append(mIoU)
                    print("Saved model at iteration %d as the best %d" % (i_iter, i))
                    best_mIoUs.sort(reverse = True)
                    best_mIoUs = best_mIoUs[:5]
                    break


        # Save for tensorboardx
        writer.add_scalar('loss', loss, i_iter)


    writer.close()



if __name__ == '__main__':
    main()
