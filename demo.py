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
from network.gated_c3d import C3dDataNetwork
from bandit_net import ContextualBanditNet, ManualController
from nnsearch.pytorch.checkpoint import CheckpointManager
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
import random
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

def draw_image(canvas, image):
    img_h, img_w, _ = image.shape
    canvas_h, canvas_w, _ = canvas.shape
    margin_h = canvas_h - img_h
    margin_w = canvas_w - img_w
    canvas[margin_h:, margin_w:, :] = image.copy()

def draw_gauge(canvas):
    h, w, c = canvas.shape
    # draw a gauge on top left corner
    radius = 50
    center = (radius * 2, radius * 2)
    center_x, center_y = center
    axes = (radius, radius)
    angle = 0
    startAngle = currentAngle = 135
    endAngle = 405
    thickness = 5
    margin = 5
    # 180 degrees
    while currentAngle < endAngle + 1:
        # add color strips
        x = (currentAngle - 180) / 180
        color = (0, 255 * (1 - x), 255 * x)

        # add ticks
        if currentAngle % ((endAngle - startAngle) // 10) == 0:
            x1 = center_x + (radius - margin) * math.cos((currentAngle - 360) *np.pi / 180.0)
            y1 = center_y + (radius - margin) * math.sin((currentAngle - 360) *np.pi / 180.0)
            x2 = center_x + (radius - 0) * math.cos((currentAngle - 360) *np.pi / 180.0)
            y2 = center_y + (radius - 0) * math.sin((currentAngle - 360) *np.pi / 180.0)
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), thickness=2)

        if currentAngle % ((endAngle - startAngle) // 2) == 0:
            u_tick = str((currentAngle - startAngle) // ((endAngle - startAngle) // 10) / 10)
            x_tick = center_x - 2.5 * margin + (radius + 6.5 * margin) * math.cos((currentAngle - 360) * np.pi / 180.0)
            y_tick = center_y + (radius + 1 * margin) * math.sin((currentAngle - 360) * np.pi / 180.0)
            cv2.putText(canvas, u_tick, (int(x_tick), int(y_tick)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.ellipse(canvas, center, axes, angle, currentAngle, currentAngle + 1, color, thickness)

        currentAngle += 1

    cv2.putText(canvas, "Throttle Meter", (int(radius * 0.8), int(radius * 3.2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
    # cv2.circle(canvas, (center_y, center_x), radius, (0,0,255), 7)


def draw_pointer(canvas, u):
    h, w, c = canvas.shape
    radius = 50
    center = (radius * 2, radius * 2)
    center_x, center_y = center
    margin = 20
    # compute angle from u
    currentAngle = 135 + 270 * u
    x1 = center_x + (radius - margin) * math.cos((currentAngle - 360) * np.pi / 180.0)
    y1 = center_y + (radius - margin) * math.sin((currentAngle - 360) * np.pi / 180.0)
    cv2.line(canvas, center, (int(x1), int(y1)), (0,0,0), 2)


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


def gesture_net_demo():
    pretrained_weights = "ckpt/cpm_r3_model_epoch1540.pth"
    full_net = GestureNet(num_refinement=3, weights_file=pretrained_weights)
    full_net.eval()

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
        heatmap_images = stack_heatmaps(batch_heatmap[b])  # clip_length images
        for i, heatmap_image in enumerate(heatmap_images):
            img = seq[i]
            loc_img = i % cols + 2 * (i // cols) * cols + 1
            loc_heatmap = i % cols + (2 * (i // cols) + 1) * cols + 1
            ax1 = fig.add_subplot(rows, cols, loc_img)
            ax1.set_title("Frame {}".format(i), fontsize=5 + 2 * 16 // total_num)
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

def throttle_demo():

    run_prediction = True
    cuda = True
    use_controller = False
    use_fixed_rule_controller = True

    def latest_checkpoints(directory, name):
        return glob.glob(os.path.join(directory, "{}_*.pkl.latest".format(name)))

    def get_class_names():
        label_file = "dataloaders/20bn-jester_labels.txt"
        label_dict = {}
        with open(label_file, 'r') as f:
            for line in f:
                idx, name = line.rstrip().split(" ", 1)
                label_dict[int(idx) - 1] = name
        return label_dict

    label_to_class = get_class_names()

    pretrained_c3d_file = latest_checkpoints("ckpt/gated_raw_c3d/", "model")[0]
    pretrained_controller_file = latest_checkpoints("ckpt/controller/", "controller_network")[0]

    checkpoint_mgr = CheckpointManager(output=".", input=".")
    # gated network
    net = C3dDataNetwork()
    checkpoint_mgr.load_parameters(pretrained_c3d_file, net, strict=True)
    net.eval()
    # print(before)
    # create a hard fixed-rule controller
    fixed_controller = ManualController()
    # controller network
    controller = ContextualBanditNet()

    checkpoint_mgr.load_parameters(pretrained_controller_file, controller, strict=True)

    controller.eval()

    if cuda:
        net = net.cuda(1)
        controller = controller.cuda(1)
        controller._us = controller._us.cuda(1)
    cam = cv2.VideoCapture(0)
    buffer = []
    canvas = np.ones((600, 368, 3), dtype=np.uint8) * 255
    draw_gauge(canvas)
    predicted_classes = ''
    u_val = 0.0
    score= 0.0
    while True:
        tmp_canvas = canvas.copy()
        _, oriImg = cam.read()
        test_img = preprocess_cam_frame(oriImg, 368)
        input = (test_img.astype(np.float32) / 255.).transpose((2, 0, 1))

        img_tensor = torch.from_numpy(input)
        buffer.append(img_tensor)
        if run_prediction and len(buffer) == 16:
            # prep the input tensor (1, 3, 16, 368, 368)
            seq_input = torch.stack(buffer, 1).unsqueeze_(0)
            if cuda:
                seq_input = seq_input.cuda(1)
            # print(seq_input.size())
            if use_controller:
                # go through controller first
                states = controller(seq_input)
                a = torch.argmax(states, 1)
                u = torch.take(controller._us, a)

            elif use_fixed_rule_controller:
                u = fixed_controller.get_utilization().cuda(1)
            else:
                u = torch.tensor([1.0]).cuda(1) if cuda else torch.tensor([1.0])

            print(u.item())
            yhat, _ = net(seq_input, u)
            # print(yhat)
            probs = torch.nn.Softmax(dim=1)(yhat)
            scores, predicted = torch.max(probs, 1)
            predicted = predicted.detach().cpu().numpy()
            score = scores.detach().cpu().numpy()[0]
            predicted_classes = [label_to_class[idx] for idx in predicted][0]
            u_val = u.item()
            fixed_controller.add_to_history(predicted_classes)
            # print(scores)
            # print(predicted_classes)
            # display text on the output

            # pop the first frame
            # buffer.pop(0)
            buffer = []
            # print(buffer)

        # draw = test_img
        cv2.putText(tmp_canvas, predicted_classes, (368 * 3 // 5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
            
        cv2.putText(tmp_canvas, "Score: {:.2f}".format(score), (368 * 3 // 5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
        draw_image(tmp_canvas, test_img)

        # u_val = random.uniform(0, 1)
        draw_pointer(tmp_canvas, u_val)
        cv2.imshow('demo', tmp_canvas)
        # cv2.waitKey(1)
        if cv2.waitKey(100) == ord('q'): break

    # # test
    # inputs = torch.randn((1, 3, 16, 368, 368)).cuda()
    # # go through controller first
    # states = controller(inputs)
    # a = torch.argmax(states, 1)
    # u = torch.take(controller._us.cuda(), a)
    # yhat, _ = net(inputs, u)
    # _, predicted = torch.max(yhat.data, 1)
    # predicted = predicted.cpu().numpy()
    # predicted_classes = [label_to_class[idx] for idx in predicted]
    # print(predicted_classes)



if __name__ == "__main__":
    throttle_demo()




