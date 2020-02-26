import os
import glob
import shutil
import contextlib
import numpy as np
from PIL import Image
import cv2
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import nnsearch.pytorch.gated.strategy as strategy
import logging
log = logging.getLogger( __name__ )
# for drawing the limbs
edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],
         [10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
colors = [cv2.cvtColor(np.uint8([[[179 * i/float(len(edges)), 179, 179]]]),cv2.COLOR_HSV2BGR)[0, 0] for i in range(len(edges))]
colors = [(int(color[0]), int(color[1]), int(color[2])) for color in colors]
def uniform_gate(umin=0):
    def f(inputs, labels):
        # return Variable( torch.rand(inputs.size(0), 1).type_as(inputs) )
        # umin = 0
        r = 1.0 - umin
        device = inputs.device
        return umin + r * torch.rand(inputs.size(0), device=device).type_as(inputs)

    return f


def constant_gate( u ):
  def f( inputs, labels ):
    # return Variable( (u * torch.ones(inputs.size(0), 1)).type_as(inputs) )
    device = inputs.device
    return (u * torch.ones(inputs.size(0), device=device)).type_as(inputs)
  return f


def penalty_fn(G, u):
    return (1 - u) * G





### save, load, checkpoint
def model_file(directory, epoch, suffix=""):
    filename = "model_{}.pkl{}".format(epoch, suffix)
    return os.path.join(directory, filename)


def latest_checkpoints(directory, prefix="model"):
    return glob.glob(os.path.join(directory, prefix + "_*.pkl.latest"))

def save_model(network, output, elapsed_epochs, force_persist=False):
    if not os.path.exists(output):
        os.mkdir(output)
    # Save current model to tmp name
    with open(model_file(output, elapsed_epochs, ".tmp"), "wb") as fout:
        if isinstance(network, torch.nn.DataParallel):
            # See: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/19
            torch.save(network.module.state_dict(), fout)
        else:
            torch.save(network.state_dict(), fout)
    # Remove previous ".latest" checkpoints
    for f in latest_checkpoints(output):
        with contextlib.suppress(FileNotFoundError):
            os.remove(f)
    # Move tmp file to latest
    os.rename(model_file(output, elapsed_epochs, ".tmp"),
              model_file(output, elapsed_epochs, ".latest"))
    checkpoint_interval = 5
    if force_persist or (elapsed_epochs % checkpoint_interval == 0):
        shutil.copy2(model_file(output, elapsed_epochs, ".latest"),
                     model_file(output, elapsed_epochs))


def checkpoint(network, output_dir, elapsed_epochs, learner, force_eval=False):
    save_model(network, output_dir, elapsed_epochs, force_persist=force_eval)
    # if force_eval or (elapsed_epochs % args.checkpoint_interval == 0):
    #     evaluate(elapsed_epochs, learner)
    # if args.aws_sync is not None:
    #     os.system(args.aws_sync)


def load_model(self, state_dict, load_gate=True, strict=True):
    def is_gate_param(k):
        return k.startswith("gate.")

    own_state = self.state_dict()

    # # remove bn keys
    # bn_keys = ["fn." + str(i) for i in [5,9,12,16,19,23,26]]
    # for name, param in state_dict.copy().items():
    #     for bn_key in bn_keys:
    #         if bn_key in name:
    #             del state_dict[name]
    # # change keys
    # own_keys = ["fn." + str(i) for i in [7,9,12,14,17,19,22,24,26]] # own dict
    # saved_keys = ["fn." + str(i) for i in [8, 11, 15, 18, 22, 25, 29, 31, 33]]
    # map_keys = {"fn." + str(v) : own_keys[i] for i, v in enumerate([8,11,15,18,22,25,29,31,33])}

    # for name, param in state_dict.copy().items():
    #     for saved_key in saved_keys:
    #         if saved_key in name:
    #             state_dict[name.replace(saved_key, map_keys[saved_key])] = state_dict[name]
    #             del state_dict[name]

    for name, param in state_dict.items():
        if name in own_state:
            log.verbose("Load %s", name)
            if not load_gate and is_gate_param(name):
                log.verbose("Skipping gate module")
                continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                if strict:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
                else:
                    print("skip {}, dimension mismatch: {} in model, {} in checkpoint.".format(name, own_state[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if not load_gate:
            missing = [k for k in missing if not is_gate_param(k)]
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

def make_sequentialGate(dict_stages, gate_during_eval=True):
    gate_modules = []
    for key, block_stages in dict_stages.items():
        if key == "refinement":
            continue
        for stage in block_stages:
            if stage.name in ["dw_conv", "conv", "fc"]:
                if stage.ncomponents > 1:
                    for _ in range(stage.nlayers):
                        # count = strategy.PlusOneCount(strategy.UniformCount(stage.ncomponents - 1)
                        # the count is random
                        # count = strategy.UniformCount(stage.ncomponents)

                        # use all components
                        # count = strategy.ConstantCount(stage.ncomponents, stage.ncomponents)

                        # the count is computed from u
                        count = strategy.ProportionToCount(0, stage.ncomponents)
                        gate_modules.append(strategy.NestedCountFromUGate(stage.ncomponents, count, gate_during_eval=gate_during_eval))
                        if stage.name == "dw_conv":
                            # count = strategy.PlusOneCount(strategy.UniformCount(stage.ncomponents - 1))
                            # count = strategy.UniformCount(stage.ncomponents)
                            # count = strategy.ConstantCount(stage.ncomponents, stage.ncomponents)
                            # the count is computed from u
                            count = strategy.ProportionToCount(0, stage.ncomponents)
                            gate_modules.append(strategy.NestedCountFromUGate(stage.ncomponents, count, gate_during_eval=gate_during_eval))

    # test
    # gate_modules.append(strategy.NestedCountGate(20, strategy.UniformCount(20)))
    return strategy.SequentialGate(gate_modules)


def get_kpts(map_6, img_h = 368.0, img_w = 368.0, t = 0.01):

    # map_6 (21,45,45)
    kpts = []
    # for m in map_6[1:]:
    for m in map_6:
        h, w = np.unravel_index(m.argmax(), m.shape)
        score = np.amax(m)
        # print(score)
        if score > t:
            x = int(w * img_w / m.shape[1])
            y = int(h * img_h / m.shape[0])
        else:
            x, y = -1, -1
        kpts.append([x,y])
    return kpts


def draw_paint(im, kpts, image_path=None, gt_kpts=None, draw_edges=True, show=False, offsets=None):
    # first need copy the image !!! Or it won't draw.
    im = im.copy()
    # draw points
    if offsets:
        for i in range(len(kpts)):
            if kpts[i][0] > -1 and kpts[i][1] > -1:
                kpts[i][0] -= offsets[0]
                kpts[i][1] -= offsets[1]
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=2, thickness=-1, color=(0, 0, 255))
    if gt_kpts:
        for k in gt_kpts:
            x = k[0]
            y = k[1]
            if x > -1 and y > -1:
                cv2.circle(im, (x, y), radius=2, thickness=-1, color=(0, 255, 0))
    # draw lines
    if draw_edges:
        for i, edge in enumerate(edges):
            s, t = edge
            if kpts[s][0] > -1 and kpts[s][1] > -1 and kpts[t][0] > -1 and kpts[t][1] > -1:
                cv2.line(im, tuple(kpts[s]), tuple(kpts[t]), color=colors[i], thickness=2)
    if show:
        cv2.imshow(image_path, im)
        cv2.waitKey(0)
    return im
    # cv2.imwrite('test_example.png', im)

def image_test(net, image_path, gated=False):

    # frame = Image.open(image_path)
    # # w, h, rgb
    # frame = frame.resize((368, 368))
    # frame_copy = np.array(frame)
    # frame_copy = frame_copy[:,:,::-1]
    # frame = transforms.ToTensor()(frame)
    # frame.unsqueeze_(0)
    # h, w, bgr
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (368, 368))
    frame_copy = frame
    input = (frame[:,:,::-1] / 255.).astype(np.float32) # np.transpose(frame[:,:,::-1], (1,0,2))
    frame = transforms.ToTensor()(input)
    frame.unsqueeze_(0)

    if gated:
        u = torch.tensor(1.0)
        # right now just one stage
        pred_6, _ = net(frame, u)
        pred = pred_6[0, -1, :, :, :].cpu().detach().numpy()
    else:
        pred_6 = net(frame)
        pred = pred_6[0, -1, :, :, :].cpu().detach().numpy()
    return pred, frame_copy

