import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
import nnsearch.logging as mylog
import logging
from network.gated_cpm_mobilenet import GatedMobilenet, GatedStage
# dataset
import configparser
from dataloaders.cmu_hand_data import CMUHand
import random
import torch.nn as nn
from modules.utils import *
from src.util import *
from network.demo_model import GestureNet
import cv2
from dataloaders.dataset import VideoDataset
import numpy as np
import math
import matplotlib.pyplot as plt
def preprocess_cam_frame(oriImg, boxsize):
    # print(oriImg.shape)
    # scale = boxsize / (oriImg.shape[0] * 1.0)
    # imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((boxsize, boxsize, 3)) * 128

    img_h = oriImg.shape[0]
    img_w = oriImg.shape[1]
    shift = 100
    if img_w < boxsize:
        offset = img_w % 2
        # make the origin image be the center
        output_img[:, shift + int(boxsize / 2 - math.floor(img_w / 2)): shift + int(
            boxsize / 2 + math.floor(img_w / 2) + offset), :] = oriImg
    else:
        # crop the center of the origin image
        output_img = oriImg[int(img_h / 2 - boxsize / 2): int(img_h / 2 + boxsize / 2),
                     shift + int(img_w / 2 - boxsize / 2): shift + int(img_w / 2 + boxsize / 2), :]
    return output_img

def cam_demo():
    pretrained_weights = "ckpt/cpm_r3_model_epoch1540.pth"
    full_net = GestureNet(num_refinement=0, weights_file=pretrained_weights)
    full_net.eval()
    full_net.heatmap_net = full_net.heatmap_net.cuda()
    cam = cv2.VideoCapture(0)
    while True:
        _, oriImg = cam.read()
        test_img = preprocess_cam_frame(oriImg, 368)
        input = (test_img[:, :, ::-1] / 255.).astype(np.float32)
        img_tensor = transforms.ToTensor()(input).unsqueeze_(0).cuda()

        pred = full_net.heatmap_net.forward(img_tensor)
        final_stage_heatmaps = pred[0,-1,:,:,:].cpu().numpy()
        kpts = get_kpts(final_stage_heatmaps, t=0.1)
        draw = draw_paint(test_img, kpts, None)
        # draw = test_img
        cv2.imshow('demo', draw.astype(np.uint8))
        # cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'): break


if __name__ == "__main__":
    pretrained_weights = "ckpt/cpm_r3_model_epoch1540.pth"
    full_net = GestureNet(num_refinement=3, weights_file=pretrained_weights)
    full_net.eval()
    # # full_net.heatmap_net = full_net.heatmap_net.cuda()
    # # # test_path = 'dataset/CMUHand/hand_labels/test/crop/Berry_roof_story.flv_000053_l.jpg'
    # # test_folder = 'dataset/CMUHand/hand_labels/test/crop'
    # # # test_folder = 'dataset/CMUHand/hand_labels_synth/crop'
    # test_folder = 'dataset/20bn-jester-preprocessed/train/Swiping Left/1022'
    # image_paths = glob.glob(os.path.join(test_folder, "*"))
    # test_path = random.choice(image_paths)
    # test_path = "dataset/20bn-jester-preprocessed/train/Swiping Left/1022/00014.jpg"
    # pred, frame = image_test(full_net.heatmap_net, test_path, gated=False)
    # kpts = get_kpts(pred)
    # draw_paint(frame, kpts, image_path=os.path.basename(test_path), show=True)

    # cam_demo()

    ################# Demo video dataset #################
    clip_length = 16
    subset = ['No gesture', 'Swiping Down', 'Swiping Up', 'Swiping Left', 'Swiping Right']
    train_data = VideoDataset(dataset='20bn-jester', split='train', clip_len=clip_length, subset=subset)
    # train_dataset = DataLoader(train_data, batch_size=1, shuffle=True)

    idx = random.choice(range(len(train_data)))
    seq, label = train_data[idx]
    seq = seq.unsqueeze(0)

    outputs, heatmaps = full_net.forward(seq, torch.tensor(1.0))
    probs = nn.Softmax(dim=1)(outputs)
    preds = torch.max(probs, 1)[1]
    pred_class = train_data.class_names[preds.cpu().int()]
    gt_class = train_data.class_names[label.cpu().int()]

    batch_seq = seq.permute(0, 2, 3, 4, 1).cpu().numpy()[:, :, :, :, ::-1]
    batch_heatmap = heatmaps.permute(0, 2, 1, 3, 4).cpu().numpy()
    total_num = batch_seq.shape[0] * clip_length * 2
    rows = 4
    cols = total_num // rows
    fig = plt.figure(figsize=(cols * 1, rows * 1.2))

    for b in range(batch_seq.shape[0]):
        seq = batch_seq[b]
        heatmap_images = stack_heatmaps(batch_heatmap[b]) # clip_length images
        for i, heatmap_image in enumerate(heatmap_images):
            img = seq[i]
            loc_img = i % cols + 2 * (i // cols) * cols + 1
            loc_heatmap = i % cols + (2 * (i // cols) + 1) * cols + 1
            ax1 = fig.add_subplot(rows, cols, loc_img)
            ax1.set_title("Frame {}".format(i), fontsize=5 + 2 * 16//total_num)
            plt.imshow(img)
            ax1.xaxis.set_major_locator(plt.NullLocator())
            ax1.yaxis.set_major_locator(plt.NullLocator())
            ax2 = fig.add_subplot(rows, cols, loc_heatmap)
            ax2.set_title("Heatmap {}".format(i), fontsize=5 + 2 * 16 // total_num)
            ax2.xaxis.set_major_locator(plt.NullLocator())
            ax2.yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(heatmap_image)
    fig.suptitle("GT: '{}' Pred: '{}'".format(gt_class, pred_class), fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.show()




