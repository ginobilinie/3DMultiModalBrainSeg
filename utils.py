import torch
# import random
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.autograd import Variable
import numpy as np
import os
#import PIL
import cv2
import math



def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def make_one_hot(labels, num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x D x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x D x H x W, where C is class number. One-hot encoded.
    '''
    labels_extend=labels.clone()
    labels_extend.unsqueeze_(1)
    #labels_extend[labels_extend > num_classes] = num_classes
    one_hot = torch.cuda.FloatTensor(labels_extend.size(0), num_classes, labels_extend.size(2), labels_extend.size(3), labels_extend.size(4)).zero_()
    one_hot.scatter_(1, labels_extend, 1) #Copy 1 to one_hot at dim=1
    #target = one_hot[:, :num_classes]#ignore the ignored class
    return one_hot



def one_hot(labels, num_classes=4):
    labels = labels.data.cpu().numpy()
    one_hot = np.zeros((labels.shape[0], num_classes, labels.shape[1], labels.shape[2],labels.shape[3]), dtype=labels.dtype)
    # handle ignore labels
    for class_id in range(num_classes):
        one_hot[:, class_id,...] = (labels==class_id)
    return torch.FloatTensor(one_hot)

def image_show(name, image, resize=5):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 3D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor) * \
           (1 - abs(og[2] - center) / factor)

    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size),
                      dtype=np.float64)
    f = math.ceil(kernel_size / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                weight[0, 0, i, j, k] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c)) *  (1 - math.fabs(k / f - c))
    #weight[range(in_channels), range(out_channels), :, :, :] = filt
    for c in range(1, in_channels):
        weight[c, 0, :, :, :] = weight[0, 0, :, :, :]
    return torch.from_numpy(weight).float()

def fill_up_weights(up):
    w = up.weight.data
    #print (w)
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            for k in range(w.size(4)):
                w[0, 0, i, j, k] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))* (1 - math.fabs(k / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :,:] = w[0, 0, :, :,:]
    #print (w)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter, lr_S, num_epoch):
    lr = lr_poly(lr_S, i_iter, num_epoch, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    #if len(optimizer.param_groups) > 1 :
    #    optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter, lr_D, num_epoch):
    lr = lr_poly(lr_D, i_iter, num_epoch, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    #if len(optimizer.param_groups) > 1 :
     #   optimizer.param_groups[1]['lr'] = lr * 10

# def fill_up_weights(up):
#     w = up.weight.data
#     f = math.ceil(w.size(2) / 2)
#     c = (2 * f - 1 - f % 2) / (2. * f)
#     for i in range(w.size(2)):
#         for j in range(w.size(3)):
#             for k in range(w.size(4)):
#                 w[0, 0, i, j, k] = \
#                     (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c)) *  (1 - math.fabs(k / f - c))
#     for c in range(1, w.size(0)):
#         w[c, 0, :, :, :] = w[0, 0, :, :, :]
#     print (w)
