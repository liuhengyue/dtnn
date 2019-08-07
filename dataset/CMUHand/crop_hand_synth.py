"""
Process CMU Hand dataset to get cropped hand datasets.

"""
import os
import numpy as np
import json
import glob
from PIL import Image

savd_dir = 'hand_labels_synth/crop/'
new_label_dir = 'hand_labels_synth/crop_label/'
if not os.path.exists(savd_dir):
    os.mkdir(savd_dir)
if not os.path.exists(new_label_dir):
    os.mkdir(new_label_dir)

dirs = ["synth1", "synth2", "synth3", "synth4"]
for data in dirs:
    query_path = os.path.join('hand_labels_synth', data, '*.jpg')
    # imgs = os.listdir('hand_labels_synth/' + data)
    imgs = glob.glob(query_path)
    for img in imgs:

        data_dir = img
        label_dir = data_dir.replace("jpg", "json")
        img_basename = os.path.basename(img).split(".")[0]

        dat = json.load(open(label_dir))
        pts = np.array(dat['hand_pts'])

        # for synthesized dataset, if it is not visible, the coordinates are also zero
        vis_pts = pts[np.where(pts[:, 2] != 0.0)]
        xmin = min(vis_pts[:, 0])
        xmax = max(vis_pts[:, 0])
        ymin = min(vis_pts[:, 1])
        ymax = max(vis_pts[:, 1])

        B = max(xmax - xmin, ymax - ymin)
        # B is the maximum dimension of the tightest bounding box
        width = 3.0 * B     # This is based on the paper

        # the center of hand box can be
        center = ((xmax + xmin) / 2, (ymax + ymin) / 2)
        # get image size
        im = Image.open(data_dir)
        w, h = im.size
        # random shift the center
        shifts = np.random.randint(- B / 2, B / 2,  size=2)
        center += shifts
        x1, y1 = max(0, center[0] - width / 2.), max(0, center[1] - width / 2.)
        x2, y2 = min(w, center[0] + width / 2.), min(h, center[1] + width / 2.)
        hand_box = [[x1, y1],
                    [x2, y2]]
        hand_box = np.array(hand_box)
        # the crop may not be square
        im = im.crop((hand_box[0, 0], hand_box[0, 1], hand_box[1, 0], hand_box[1, 1]))
        im = im.resize((368, 368))
        save_img_path = os.path.join(savd_dir, data + "_" + os.path.basename(img))
        im.save(save_img_path)  # save cropped image

        pts[:, :2] = pts[:, :2] - hand_box[0, :]
        # different ratio
        box_w = x2 - x1
        box_h = y2 - y1
        pts[:, 0] = pts[:, 0] * 368 / box_w
        pts[:, 1] = pts[:, 1] * 368 / box_h
        lbl = pts.tolist()

        label_dict = {}
        label_dict['hand_pts_crop'] = lbl
        json.dump(label_dict, open(new_label_dir + data + "_" + img_basename + '.json', 'w'))






