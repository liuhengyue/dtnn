import os
import matplotlib
import numpy as np
import scipy.misc
import cv2

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image


def heatmap_image(img, label, save_dir='../data_loader/img/heat.jpg'):
    """
    draw heat map of each joint
    :param img:             a PIL Image
    :param heatmap          type: numpy     size: 21 * 45 * 45
    :return:
    """

    im_size = 128

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
    os.remove('tmp.jpg')




def save_images(images, label_map, predict_heatmaps, step, epoch, imgs, save_dir='ckpt/'):
    """
    :param images: Batch_size   * 3 *   368 * 368
    :param label_map:                       Batch_size   * joints *   45 * 45
    :param predict_heatmaps:    4D Tensor    Batch_size   * joints *   45 * 45
    :return:
    """
    if not os.path.exists(save_dir + 'epoch' + str(epoch)):
        os.mkdir(save_dir + 'epoch' + str(epoch))

    output_size = 100
    actual_size = 90
    margin = (output_size - actual_size) // 2
    num_patches = 3
    for b in range(label_map.shape[0]):                     # for each batch (person)
        output = np.ones((output_size * num_patches, output_size, 3))           # cd .. temporal save a single image
        img_name = os.path.basename(imgs[b])
        # seq = imgs[b].split('/')[-2]                     # sequence name 001L0

        pre = np.zeros((actual_size, actual_size))  #
        gth = np.zeros((actual_size, actual_size))
        # im = imgs[b].split('/')[-1][1:5]  # image name 0005
        img_data = np.asarray(images[b, :, :, :].data)
        img_data = np.transpose(img_data, (1, 2, 0))
        img_data = cv2.resize(img_data, dsize=(actual_size, actual_size))
        output[margin : actual_size + margin, margin : actual_size + margin, :] = img_data * 255
        for i in range(21):
            heatmap = np.asarray(predict_heatmaps[b, i, :, :].data)
            gt_heatmap = np.asarray(label_map[b, i, :, :].data)
            heatmap = cv2.resize(heatmap, dsize=(actual_size, actual_size))
            gt_heatmap = cv2.resize(gt_heatmap, dsize=(actual_size, actual_size))
            pre += heatmap  # 2D
            gth += gt_heatmap  # 2D

            output[margin * 2 + actual_size : actual_size * 2 + margin * 2, margin : actual_size + margin, :] = np.stack((gth, gth, gth), axis=-1)
            output[margin * 3 + actual_size * 2 : actual_size * 3 + margin * 3, margin : actual_size + margin, :] = np.stack((pre, pre, pre), axis=-1)

        # convert to uint8
        output[margin * 2 + actual_size: actual_size * 2 + margin * 2, margin: actual_size + margin, :] = \
            255 * output[margin * 2 + actual_size: actual_size * 2 + margin * 2, margin: actual_size + margin, :] / \
            (np.max(output[margin * 2 + actual_size: actual_size * 2 + margin * 2, margin: actual_size + margin, :]) + 1e-16)
        # print(np.max(output[margin * 2 + actual_size: actual_size * 2 + margin * 2, margin: actual_size + margin, :]))
        output[margin * 3 + actual_size * 2: actual_size * 3 + margin * 3, margin: actual_size + margin, :] = \
            255 * output[margin * 3 + actual_size * 2 : actual_size * 3 + margin * 3, margin : actual_size + margin, :] / \
            (np.max(output[margin * 3 + actual_size * 2: actual_size * 3 + margin * 3, margin: actual_size + margin, :]) + 1e-16)
        # scipy.misc.imsave(save_dir + 'epoch'+str(epoch) + '/s'+str(step)
        #                   + '_b' + str(b) + '_' + seq + '_' + im + '.jpg', output)
        # patch = output[margin * 2 + actual_size: actual_size * 2 + margin * 2, margin: actual_size + margin, :]
        # plt.imshow(patch)
        # plt.show()

        scipy.misc.imsave(save_dir + 'epoch' + str(epoch) + '/s' + str(step)
                          + '_b' + str(b) + '_' + img_name, output)


def stack_heatmaps(heatmaps, output_size=90):
    """
    :param heatmaps:    4D Tensor    Batch_size   * joints *   45 * 45
    :return:
    """
    output = np.zeros((heatmaps.shape[0], output_size, output_size))
    for b in range(heatmaps.shape[0]):                     # for each batch (person)

        pre = np.zeros((output_size, output_size))  #

        for i in range(21):
            heatmap = np.asarray(heatmaps[b, i, :, :].data)
            heatmap = cv2.resize(heatmap, dsize=(output_size, output_size))
            pre += heatmap  # 2D
        output[b] = pre

    return output



def PCK(predict, target, label_size=45, sigma=0.04):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than  the value is 1, otherwise it is 0
    :param predict:         3D numpy       21 * 45 * 45
    :param target:          3D numpy       21 * 45 * 45
    :param label_size:
    :param sigma:
    :return: 0/21, 1/21, ...
    """
    pck = 0
    for i in range(predict.shape[0]):       # 21

        pre_x, pre_y = np.where(predict[i, :, :] == np.max(predict[i, :, :]))
        tar_x, tar_y = np.where(target[i, :, :] == np.max(target[i, :, :]))

        dis = np.sqrt((pre_x[0] - tar_x[0])**2 + (pre_y[0] - tar_y[0])**2)
        if dis < sigma * label_size:
            pck += 1

    return pck / 21.0


