"""
This is the data loader of Hand Pose datasets to train or test CPM model

There are four variables in one item
image   Type: Tensor    Size: 3 * 368 * 368
label   Type: Tensor    Size: 21 * 45 * 45
center  Type: Tensor    Size: 3 * 368 * 368
name    Type:  str

The data is organized in the following style

----data                        This is the folder name like train or test
------------L0001.jpg
------------L0007.jpg
------------ ....
--------001L1
------------L0100.jpg
------------L0107.jpg
------------ ....
-------- .....

----label                        This is the folder name like train or test
--------001L0.json               This is one sequence of images
--------001L1.json
------------ ....

To have a better understanding, you can view ../dataset in this repo
"""

import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import json
import imageio
import glob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from src.util import *

def activator(images, augmenter, parents, default):
    return default if augmenter.__class__.__name__ in ["Sequential", "Affine", "Fliplr", "Crop", "GaussianBlur"] \
        else False

class CMUHand(Dataset):

    def __init__(self, data_dir, label_dir, joints=21,  sigma=0.5, mode="train",
                 augment=True, negative_augment=True):
        super(CMUHand, self).__init__()
        assert mode in ["train", "test"], "wrong mode for the dataset"
        self.mode = mode

        self.height = 368
        self.width = 368

        self.data_dir = data_dir
        self.label_dir = label_dir

        self.joints = joints  # 21 heat maps
        self.sigma = sigma  # gaussian center heat map sigma

        self.images_dir = []
        self.annotation_dir = []
        self.augment = augment
        self.negative_augment = negative_augment # for supplimentary negative class training
        self.gen_imgs_dir()
        self.aug = self.aug_generator()


    def gen_imgs_dir(self):
        """
        get absolute directory of all images
        :return:
        """
        if isinstance(self.data_dir, list):
            for img_dir in self.data_dir:
                imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
                # glob order is not guaranteed on Linux not sure why
                imgs.sort()
                self.images_dir.extend(imgs) # absolute dir
            if self.mode == "train":
                for anno_dir in self.label_dir:
                    annos = glob.glob(os.path.join(anno_dir, "*.json"))
                    annos.sort()
                    self.annotation_dir.extend(annos)
        else:
            print("Loading from " + self.data_dir)
            imgs = glob.glob(os.path.join(self.data_dir, "*.jpg"))
            imgs.sort()
            self.images_dir.extend(imgs) # absolute dir
            if self.mode == "train":
                annos = glob.glob(os.path.join(self.label_dir, "*.json"))
                annos.sort()
                self.annotation_dir.extend(annos)
        # add negative images
        self.add_negative_dir()
        # make sure everything match
        if self.mode == "train":
            assert len(self.images_dir) == len(self.annotation_dir), \
                "number of images [{}] and annotations [{}] mismatch.".format(len(self.images_dir), len(self.annotation_dir))

        print('total number of image is ' + str(len(self.images_dir)))

    def add_negative_dir(self):
        negative_dir = 'dataset/20bn-jester-preprocessed/train/No gesture'
        # grab all images in the sub-folders
        glob_query = os.path.join(negative_dir, "**/*.jpg")
        imgs = glob.glob(glob_query, recursive=True)
        print("number of negative images: {}".format(len(imgs)))
        # positive : negative ratio 1:2
        positive_num = len(self.images_dir)
        negative_num = 2 * positive_num
        negative_num = min(negative_num, len(imgs))
        random.shuffle(imgs)
        imgs = imgs[:negative_num]
        # add to imgs dir
        self.images_dir.extend(imgs)
        print("Use number of negative images: {}".format(len(imgs)))
        if self.mode == "train":
            # add artificial annotations
            self.annotation_dir.extend(["negative"] * len(imgs))

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        images          3D Tensor      3                *   height(368)      *   weight(368)
        label_map       3D Tensor      joint            *   label_size(45)   *   label_size(45)
        center_map      3D Tensor      1                *   height(368)      *   weight(368)
        """

        # get image
        img = self.images_dir[idx]              # '.../001L0/L0005.jpg' note: not necessarily true for windows
        im = Image.open(img)                # read image

        w, h, c = np.asarray(im).shape      # weight 256 * height 256 * 3
        ratio_x = self.width / float(w)
        ratio_y = self.height / float(h)    # 368 / 368 = 1
        im = im.resize((self.width, self.height))                       # unit8      weight 368 * height 368 * 3

        image = transforms.ToTensor()(im)   # 3D Byte Tensor  3 * height 368 * weight 368


        # generate the Gaussian heat map (unused right now)
        # center_map = self.genCenterMap(x=self.width / 2.0, y=self.height / 2.0, sigma=21,
        #                                size_w=self.width, size_h=self.height)
        # center_map = torch.from_numpy(center_map)
        # get label map
        if self.mode == "train":
            label_size = self.width // 8 - 1         # 45
            label_path = self.annotation_dir[idx]
            # debug
            # assert os.path.basename(img).split('.')[:-1] == os.path.basename(label_path).split('.')[:-1], \
            # "Mismatch on image [{}] and annotation [{}]".format(img, label_path)
            if label_path == "negative":
                label = self.joints * [[0, 0, 0]]
            else:
                labels = json.load(open(label_path))

                label = labels['hand_pts_crop']         # 0005  list       21 * 2

            # augmentations

            aug = self.aug.to_deterministic() if self.augment else None

            lbl, vis_mask = self.genLabelMap(label, label_size=label_size,joints=self.joints,
                                             ratio_x=ratio_x, ratio_y=ratio_y, augment=aug)
            if self.augment:
                img_aug = aug.augment_image(np.array(im))
                # heatmap_aug = aug.augment_images(lbl, hooks = ia.HooksImages(activator=activator))
                # heatmap_aug = aug.augment_image(np.transpose(lbl,(1,2,0)), hooks=ia.HooksImages(activator=activator))
                # lbl: c,h,w; np im: w, h, 3
                # img_aug, heatmap_aug = aug(image=np.asarray(im), heatmaps=np.expand_dims(lbl, axis=0).astype(np.float32))
                image = transforms.ToTensor()(img_aug.copy())
                label_maps = torch.from_numpy(lbl)
            else:
                image = transforms.ToTensor()(im)
                label_maps = torch.from_numpy(lbl)
            visible_mask = torch.from_numpy(vis_mask)

            return image.float(), label_maps.float(), visible_mask.byte(), img

        
        # else return without labels
        return image.float(), img

    def genCenterMap(self, x, y, sigma, size_w, size_h):
        """
        generate Gaussian heat map
        :param x: center point
        :param y: center point
        :param sigma:
        :param size_w: image width
        :param size_h: image height
        :return:            numpy           w * h
        """
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)  # numpy 2d

    def genLabelMap(self, label, label_size, joints, ratio_x, ratio_y, augment=None):
        """
        generate label heat map
        :param label:               list            21 * 3 (changed to 3 dim with the 3rd as visibility)
        visibility = -1 means invisable which do not contribute to loss;
        visibility = 0 means using negative classes which contribute to loss;
        visibility = 1 means visable keypoints which contribute to loss.
        :param label_size:          int             45
        :param joints:              int             21
        :param ratio_x:             float           1.4375
        :param ratio_y:             float           1.4375
        :return:  heatmap           numpy           joints * boxsize/stride * boxsize/stride
        """
        # initialize
        label_maps = np.zeros((joints, label_size, label_size))
        background = np.zeros((label_size, label_size))
        visible = np.ones((joints))
        # each joint
        for i in range(len(label)):
            lbl = label[i]                      # [x, y] for real, [x, y, vis] for synth
            # CMU read or synth data
            if (len(lbl) == 3 and lbl[2] > 0) or len(lbl) == 2:
                x = lbl[0] * ratio_x / 8.0          # modify the label
                y = lbl[1] * ratio_y / 8.0
                if augment is not None:
                    kpt = KeypointsOnImage([Keypoint(x, y)], shape=(label_size, label_size))
                    kpt_aug = augment.augment_keypoints(kpt)
                    x, y = kpt_aug.keypoints[0].x, kpt_aug.keypoints[0].y

                if x > -1 and x < 45 and y > -1 and y < 45:
                    heatmap = self.genCenterMap(y, x, sigma=self.sigma, size_w=label_size, size_h=label_size)  # numpy
                    background += heatmap               # numpy
                    label_maps[i, :, :] = np.transpose(heatmap)  # !!!
                # if augmented keypoints out of boundary
                else:
                    visible[i] = 0
            # for negative class
            elif lbl[2] == 0:
                pass
            # skip invisible keypoints
            else:
                visible[i] = 0

        return label_maps, visible  # numpy           label_size * label_size * (joints + 1)

    def aug_generator(self):
        seq = iaa.Sequential([
            # iaa.Fliplr(0.5),  # horizontal flips
            iaa.Crop(percent=(0, 0.2)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.GaussianBlur(sigma=(0, 0.5)),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.5, 1.2), "y": (0.5, 1.2)},
                translate_percent={"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)  # apply augmenters in random order

        return seq


# test case
if __name__ == "__main__":
    data_dir = 'dataset/CMUHand/hand_labels/train/crop'
    label_dir = 'dataset/CMUHand/hand_labels/train/crop_label'
    data = CMUHand(data_dir=data_dir, label_dir=label_dir)

    img, label, center, name = data[-1]
    print('dataset info ...')
    print(img.shape)         # 3D Tensor 3 * 368 * 368
    print(label.shape)       # 3D Tensor 21 * 45 * 45
    print(center.shape)      # 2D Tensor 368 * 368
    print(name)              # str   ../dataset/train_data/001L0/L0461.jpg

    # ***************** draw label map *****************
    print('draw label map ...')
    lab = np.asarray(label)
    out_labels = np.zeros((45, 45), dtype=np.float32)
    for i in range(21):
        out_labels += lab[i, :, :]
    out_labels = (np.where(out_labels > 1.0, 1.0, out_labels) * 255).astype(np.uint8)
    imageio.imwrite('visualization/cmu_label.jpg', out_labels)

    # ***************** draw image *****************
    print('draw image ')
    im_size = 368
    target = Image.new('RGB', (im_size, im_size))
    img = transforms.ToPILImage()(img)
    img.save('visualization/cmu_img.jpg')

    heatmap = np.asarray(label[0, :, :])

    im = Image.open('visualization/cmu_img.jpg')

    heatmap_image(img, lab, save_dir='visualization/cmu_heat.jpg')




