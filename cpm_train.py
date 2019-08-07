"""


"""

# from data_loader.uci_hand_data import UCIHandPoseDataset as Mydata
from dataloaders.cmu_hand_data import CMUHand as Mydata
from network.cpm import CPM
from network.cpm_mobilenet import CPM_MobileNet
from src.util import *
import scipy.misc

import os
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.nn as nn
import configparser

from torch.autograd import Variable
from torch.utils.data import DataLoader


# multi-GPU
device_ids = [0, 1]

# *********************** hyper parameter  ***********************

config = configparser.ConfigParser()
config.read('conf.text')
train_data_dir = config.get('data', 'train_data_dir')
train_label_dir = config.get('data', 'train_label_dir')
train_synth_data_dir = config.get('data', 'train_synth_data_dir')
train_synth_label_dir = config.get('data', 'train_synth_label_dir')
save_dir = config.get('data', 'save_dir')

learning_rate = config.getfloat('training', 'learning_rate')
batch_size = config.getint('training', 'batch_size')
epochs = config.getint('training', 'epochs')
begin_epoch = config.getint('training', 'begin_epoch')

cuda = torch.cuda.is_available()

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# *********************** Build dataset ***********************
train_data = Mydata(data_dir=train_data_dir, label_dir=train_label_dir)
# train_data = Mydata(data_dir=[train_data_dir,train_synth_data_dir], label_dir=[train_label_dir, train_synth_label_dir])
print('Train dataset total number of images sequence is ----' + str(len(train_data)))

# Data Loader
train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# *********************** Build model ***********************

# net = CPM(out_c=21)
n_refine_stages = 3
net = CPM_MobileNet(n_refine_stages)
if cuda:
    print("Detected gpus.")
    net = net.cuda(device_ids[0])
    net = nn.DataParallel(net, device_ids=device_ids)


if begin_epoch > 0:
    save_path = os.path.join(save_dir, 'cpm_r' + str(n_refine_stages) + '_model_epoch' + str(begin_epoch) + '.pth')
    state_dict = torch.load(save_path, lambda storage, loc: storage)
    net.module.load_state_dict(state_dict)
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


def train():
    # *********************** initialize optimizer ***********************
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.MSELoss(reduction='sum')                       # loss function MSE average

    net.train()
    for epoch in range(begin_epoch, epochs + 1):
        print('epoch....................' + str(epoch))
        for step, (image, label_map, center_map, imgs) in enumerate(train_dataset):
            image = Variable(image.cuda() if cuda else image)                   # 4D Tensor
            # Batch_size  *  3  *  width(368)  *  height(368)

            # 4D Tensor to 5D Tensor
            label_map = torch.stack([label_map]*(n_refine_stages + 1), dim=1)
            # Batch_size  *  21 *   45  *  45
            # Batch_size  *   6 *   21  *  45  *  45
            label_map = Variable(label_map.cuda() if cuda else label_map)

            # center_map = Variable(center_map.cuda() if cuda else center_map)    # 4D Tensor
            # Batch_size  *  width(368) * height(368)

            optimizer.zero_grad()
            # pred_6 = net(image, center_map)  # 5D tensor:  batch size * stages * 21 * 45 * 45
            pred_6 = net(image)

            # ******************** calculate loss of each joints ********************
            loss = criterion(pred_6, label_map)

            # backward
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print('--step .....' + str(step))
                print('--loss ' + str(float(loss.data.item())))

            if step % 200 == 0:
                save_images(label_map[:, -1, :, :, :].cpu(), pred_6[:, -1, :, :, :].cpu(), step, epoch, imgs)

        if epoch % 5 == 0:
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(),
                           os.path.join(save_dir, 'cpm_r' + str(n_refine_stages) + '_model_epoch{:d}.pth'.format(epoch)))
            else:
                torch.save(net.state_dict(),
                           os.path.join(save_dir, 'cpm_r' + str(n_refine_stages) + '_model_epoch{:d}.pth'.format(epoch)))

    print('train done!')


if __name__ == '__main__':
    train()








