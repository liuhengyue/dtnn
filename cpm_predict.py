"""
Predict 21 key points for images without ground truth
step 1: predict label and save into json file for every image

"""

# from data_loader.uci_hand_data import UCIHandPoseDataset as Mydata
from dataloaders.cmu_hand_data import CMUHand as Mydata
from network.cpm import CPM

import configparser
import numpy as np
import os
import math
import json
from collections import OrderedDict


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2


# *********************** hyper parameter  ***********************

device_ids = [0, 1]        # multi-GPU

config = configparser.ConfigParser()
config.read('conf.text')

batch_size = config.getint('training', 'batch_size')
epochs = config.getint('training', 'epochs')
begin_epoch = config.getint('training', 'begin_epoch')

best_model = config.getint('test', 'best_model')

predict_data_dir = config.get('predict', 'predict_data_dir')
predict_label_dir = config.get('predict', 'predict_label_dir')
predict_labels_dir = config.get('predict', 'predict_labels_dir')


heatmap_dir = 'ckpt/'
cuda = torch.cuda.is_available()

sigma = 0.04

# for drawing the limbs
edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

colors = [cv2.cvtColor(np.uint8([[[179 * i/float(len(edges)), 179, 179]]]),cv2.COLOR_HSV2BGR)[0, 0] for i in range(len(edges))]
colors = [(int(color[0]), int(color[1]), int(color[2])) for color in colors]


# *********************** function ***********************
if not os.path.exists(predict_label_dir):
    os.mkdir(predict_label_dir)

if not os.path.exists(predict_labels_dir):
    os.mkdir(predict_labels_dir)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def heatmap_image(img, label,save_dir='/home/haoyum/Tdata/heat_maps/'):
    """
    draw heat map of each joint
    :param img:             a PIL Image
    :param heatmap          type: numpy     size: 21 * 45 * 45


    :return:
    """

    im_size = 64

    img = img.resize((im_size, im_size))
    x1 = 0
    x2 = im_size

    y1 = 0
    y2 = im_size

    target = Image.new('RGB', (7 * im_size, 3 * im_size))
    for i in range(21):
        heatmap = label[i, :, :]    # heat map for single one joint

        # remove white margin
        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        fig = plt.gcf()

        fig.set_size_inches(7.0 / 3, 7.0 / 3)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(heatmap)
        plt.text(10, 10, '{0}'.format(i), color='r', fontsize=24)
        plt.savefig('tmp.jpg')

        heatmap = Image.open('tmp.jpg')
        heatmap = heatmap.resize((im_size, im_size))

        img_cmb = Image.blend(img, heatmap, 0.5)

        target.paste(img_cmb, (x1, y1, x2, y2))

        x1 += im_size
        x2 += im_size

        if i == 6 or i == 13:
            x1 = 0
            x2 = im_size
            y1 += im_size
            y2 += im_size

    target.save(save_dir)
    os.system('rm tmp.jpg')

def get_kpts(map_6, img_h = 368.0, img_w = 368.0):

    # map_6 (21,45,45)
    kpts = []
    # for m in map_6[1:]:
    for m in map_6:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts

def draw_paint(im, kpts):
    # first need copy the image !!! Or it won't draw.
    im = im.copy()
    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=1, thickness=-1, color=(0, 0, 255))

    # draw lines
    
    for i, edge in enumerate(edges):
        s, t = edge
        cv2.line(im, tuple(kpts[s]), tuple(kpts[t]), color=colors[i])

    cv2.imshow('test_example', im)
    cv2.waitKey(0)
    cv2.imwrite('test_example.png', im)

def Tests_save_label(predict_heatmaps, step, imgs):
    """
    :param predict_heatmaps:    4D Tensor    batch size * 21 * 45 * 45
    :param step:
    :param imgs:                batch_size * 1
    :return:
    """
    for b in range(predict_heatmaps.shape[0]):  # for each batch (person)
        seq = imgs[b].split('/')[-2]  # sequence name 001L0
        label_dict = {}  # all image label in the same seq

        labels_list = []  # 21 points label for one image [[], [], [], .. ,[]]
        im = imgs[b].split('/')[-1][1:5]  # image name 0005

        # ****************** save image and label of 21 joints ******************
        for i in range(21):  # for each joint
            tmp_pre = np.asarray(predict_heatmaps[b, i, :, :].data)  # 2D
            #  get label of original image
            corr = np.where(tmp_pre == np.max(tmp_pre))
            x = corr[0][0] * (256.0 / 45.0)
            x = int(x)
            y = corr[1][0] * (256.0 / 45.0)
            y = int(y)
            labels_list.append([y, x])  # save img label to json

        label_dict[im] = labels_list  # save label

        # ****************** save label ******************
        save_dir_label = predict_label_dir + '/' + seq          # 101L0
        if not os.path.exists(save_dir_label):
            os.mkdir(save_dir_label)
        json.dump(label_dict, open(save_dir_label + '/' + str(step) +
                                   '_' + im + '.json', 'w'), sort_keys=True, indent=4)


# ************************************ Build dataset ************************************
test_data = Mydata(data_dir=predict_data_dir, label_dir=None, mode="test")
print('Test dataset total number of images is ----' + str(len(test_data)))

# Data Loader
test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# Build model
net = CPM(21)
if cuda:
    net = net.cuda(device_ids[0])
    net = nn.DataParallel(net, device_ids=device_ids)  # multi-Gpu

model_path = os.path.join('ckpt/model_epoch' + str(best_model)+'.pth')
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
if cuda:
    net.load_state_dict(state_dict)
else:
    # trained with DataParallel but test on cpu
    single_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        single_state_dict[name] = v
    # load params
    net.load_state_dict(single_state_dict)


# **************************************** test all images ****************************************

print('********* test data *********')
net.eval()

for step, (image, center_map, imgs) in enumerate(test_dataset):
    image_cpu = image
    image = Variable(image.cuda() if cuda else image)   # 4D Tensor
    # Batch_size  *  3  *  width(368)  *  height(368)
    center_map = Variable(center_map.cuda() if cuda else center_map)  # 4D Tensor
    # Batch_size  *  width(368) * height(368)

    pred_6 = net(image, center_map)  # 5D tensor:  batch size * stages(6) * 41 * 45 * 45

    # ****************** from heatmap to label ******************
    Tests_save_label(pred_6[:, 5, :, :, :].cpu(), step, imgs=imgs)


    # ****************** draw heat maps ******************
    for b in range(image_cpu.shape[0]):
        img = image_cpu[b, :, :, :]         # 3D Tensor
        img = transforms.ToPILImage()(img.data)        # PIL Image
        pred = pred_6[b, 5, :, :, :].cpu().detach().numpy()     # 3D Numpy

        seq = imgs[b].split('/')[-2]  # sequence name 001L0
        im = imgs[b].split('/')[-1][1:5]  # image name 0005
        if not os.path.exists(heatmap_dir + seq):
            os.mkdir(heatmap_dir+seq)
        img_dir = heatmap_dir + seq + '/' + im + '.jpg'
        heatmap_image(img, pred, save_dir=img_dir)
        # ****************** draw keypoints on image ******************
        kpts = get_kpts(pred)
        img_copy = image_cpu[b, :, :, :].permute(1, 2, 0).numpy()
        img_copy = img_copy[:, :, ::-1] * 255
        img_copy = img_copy.astype(np.uint8)
        # print(img_copy)
        # print(type(img_copy), img_copy.shape)
        # print(kpts)
        draw_paint(img_copy, kpts)
        break

    break






# ****************** merge label json file ******************

# print('merge json file ............ ')

# seqs = os.listdir(predict_label_dir)

# for seq in seqs:
#     if seq == '.DS_Store':
#         continue
#     print(seq)

#     s = os.path.join(predict_label_dir, seq)
#     steps = os.listdir(s)
#     d = {}
#     for step in steps:
#         lbl = json.load(open(s + '/' + step))
#         d = dict(d.items() + lbl.items())

#     json.dump(d, open(predict_labels_dir + '/' + seq + '.json', 'w'), sort_keys=True, indent=4)

# os.system('rm -r '+predict_label_dir)

# print('build video ......')
