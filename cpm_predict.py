"""
Predict 21 key points for images without ground truth
step 1: predict label and save into json file for every image

"""

# from data_loader.uci_hand_data import UCIHandPoseDataset as Mydata
from dataloaders.cmu_hand_data import CMUHand as Mydata
from network.cpm import CPM
from network.cpm_mobilenet import CPM_MobileNet

import configparser
import numpy as np
import os
import math
import json
from collections import OrderedDict


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2


# *********************** hyper parameter  ***********************

device_ids = [0, 1]        # multi-GPU

config = configparser.ConfigParser()
config.read('conf.text')

batch_size = config.getint('training', 'batch_size')
epochs = config.getint('training', 'epochs')
begin_epoch = config.getint('training', 'begin_epoch')
train_data_dir = config.get('data', 'train_data_dir')
train_label_dir = config.get('data', 'train_label_dir')
train_synth_data_dir = config.get('data', 'train_synth_data_dir')
train_synth_label_dir = config.get('data', 'train_synth_label_dir')
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

def heatmap_image(img, label, save_dir='visualization/heatmap.jpg'):
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


def np2heatmap(heatmap, save_dir='visualization/'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    c = heatmap.shape[0]
    for i in range(c):
        img_dir = os.path.join(save_dir, "{}.jpg".format(i))
        matplotlib.image.imsave(img_dir, heatmap[i], cmap='gray')


def get_kpts(map_6, img_h = 368.0, img_w = 368.0, t = 0.01):

    # map_6 (21,45,45)
    kpts = []
    # for m in map_6[1:]:
    for m in map_6:
        h, w = np.unravel_index(m.argmax(), m.shape)
        score = np.amax(m, axis=None)
        # score = 1
        # print(score)
        if score > t:
            x = int(w * img_w / m.shape[1])
            y = int(h * img_h / m.shape[0])
        else:
            x, y = -1, -1
        kpts.append([x,y])
    return kpts

def draw_paint(im, kpts, image_path, gt_kpts=None, draw_edges=True):
    # first need copy the image !!! Or it won't draw.
    im = im.copy()
    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=1, thickness=-1, color=(0, 0, 255))
    if gt_kpts:
        for k in gt_kpts:
            x = k[0]
            y = k[1]
            if x > -1 and y > -1:
                cv2.circle(im, (x, y), radius=1, thickness=-1, color=(0, 255, 0))
    # draw lines
    if draw_edges:
        for i, edge in enumerate(edges):
            s, t = edge
            if kpts[s][0] > -1 and kpts[s][1] > -1 and kpts[t][0] > -1 and kpts[t][1] > -1:
                cv2.line(im, tuple(kpts[s]), tuple(kpts[t]), color=colors[i])

    cv2.imshow(image_path, im)
    cv2.waitKey(0)
    # cv2.imwrite('test_example.png', im)

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

def image_test(net, image_path, draw=True):
    frame = Image.open(image_path)
    frame = frame.resize((368, 368))
    frame_copy = np.array(frame)
    frame_copy = frame_copy[:,:,::-1]
    frame = transforms.ToTensor()(frame)
    frame.unsqueeze_(0)
    frame = Variable(frame)
    pred_6 = net(frame)
    pred = pred_6[0, -1, :, :, :].cpu().detach().numpy()
    # heatmap_image(Image.open(image_path), pred)
    if draw:
        kpts = get_kpts(pred)
        draw_paint(frame_copy, kpts, image_path)
    return pred
    # heatmap_image(img, pred)






# Build model
# net = CPM(21)
n_refine_stages = 3
net = CPM_MobileNet(n_refine_stages)
model_path = os.path.join("ckpt/", 'cpm_r' + str(n_refine_stages) + '_model_epoch{:d}.pth'.format(1020))
net.load_pretrained_weights(model_path)
net = net.eval()
cuda = False
if cuda:
    net = net.cuda(device_ids[0])
    # net = nn.DataParallel(net, device_ids=device_ids)  # multi-Gpu


# state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
# print(state_dict.keys())
# print("-----")
# print(net.state_dict().keys())
# net.load_state_dict(state_dict)

# print(state_dict.keys())
# if cuda:
#     net.load_state_dict(state_dict)
# else:
#     # trained with DataParallel but test on cpu
#     single_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k.replace("module.", "") # remove `module.`
#         single_state_dict[name] = v
#     # load params
#     net.load_state_dict(single_state_dict)


# **************************************** test all images ****************************************

# print('********* test data *********')
# net.eval()
# test_path = 'dataset/CMUHand/hand_labels/test/crop/Berry_roof_story.flv_000053_l.jpg'
# test_folder = 'dataset/20bn-jester-preprocessed/val/Stop Sign/234'
# save_base = os.path.join('visualization', os.path.basename(test_folder))
# if not os.path.exists(save_base):
#     os.mkdir(save_base)
# test_file_names = os.listdir(test_folder)
# for file_name in test_file_names:
#     img_dir = os.path.join(test_folder, file_name)
#     pred = image_test(net, img_dir)
#     save_dir = os.path.join(save_base, os.path.splitext(file_name)[0])
#
#     np2heatmap(pred, save_dir=save_dir)

# **************************************** test single hand image ***********************************
import glob, random
# ************************************ Build dataset ************************************
# test_data = Mydata(data_dir=predict_data_dir, label_dir=predict_label_dir, mode="test")
train_data = Mydata(data_dir=train_data_dir, label_dir=train_label_dir, mode="train")

# train_data = Mydata(data_dir=train_synth_data_dir, label_dir=train_synth_label_dir, mode="train")
print('Train dataset total number of images is ----' + str(len(train_data)))
# print('Test dataset total number of images is ----' + str(len(test_data)))

# Data Loader
train_dataset = DataLoader(train_data, batch_size=4, shuffle=True)
# test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)
idx = random.choice(range(len(train_data)))
# idx = 0
img, label, vis_mask, name = train_data[idx]
# print(vis_mask)
# torch.set_printoptions(threshold=5000)

# check if loading correct image and labels
# for step, (image, label_map, center_map, imgs) in enumerate(train_dataset):
#     pass

########################## check the images, gt_keypoints, and prediction
from src.util import *
# label = label.unsqueeze(0)
# save_images(img.unsqueeze_(0).cpu(), label.cpu(), label.cpu(), 0, 0, [name])
# input = img.unsqueeze_(0).cuda(device_ids[0]) if cuda else img.unsqueeze_(0)
# pred_6 = net(input)
# pred = pred_6[0, -1, :, :, :]
# save_images(input.cpu(), label.unsqueeze(0).cpu(), pred.unsqueeze(0).cpu(), 0, 0, [name])
# img = img.unsqueeze(0).cpu().numpy()
# label = label.unsqueeze(0).cpu().numpy()
# save_images(img, label, label, 0, 0, [name])
#
#
# frame = Image.open(name)
# frame = frame.resize((368, 368))
# frame_copy = np.array(frame)
# frame_copy = frame_copy[:,:,::-1]
# kpts = get_kpts(pred)

frame_copy = (np.transpose(img.cpu().numpy(), (1,2,0))[:,:,::-1] * 255).astype(np.uint8)
gt_kpts = get_kpts(label.cpu().numpy())
draw_paint(frame_copy, gt_kpts, os.path.basename(name), gt_kpts=None)
# print(name)
###### GT #####
# frame = Image.open(name)
# frame = frame.resize((368, 368))
# frame_copy = np.array(frame)
# frame_copy = frame_copy[:,:,::-1]
# kpts = get_kpts(label)
# draw_paint(frame_copy, kpts, "gt")
#########
# test_folder = 'dataset/CMUHand/hand_labels/test/crop'
# test_folder = 'dataset/20bn-jester-preprocessed/train/Stop Sign/31'
# test_folder = 'dataset/20bn-jester-preprocessed/train/Swiping Down/67'
# image_paths = glob.glob(os.path.join(test_folder, "*"))
# name = random.choice(image_paths)
# image_test(net, name)
# test_path = 'dataset/CMUHand/hand_labels/train/crop/015986866_01_r.jpg'
# test_path = 'dataset/20bn-jester-preprocessed/val/Stop Sign/234/00027.jpg'
# name = "dataset/CMUHand/hand_labels/train/crop/Alexander_mouse_cat_rooster.flv_000150_r.jpg"
