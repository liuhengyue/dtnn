import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
import nnsearch.logging as mylog
import logging
import nnsearch.pytorch.gated.strategy as strategy
from nnsearch.pytorch.gated.module import GatedChainNetwork
from network.cpm_mobilenet import CPM_MobileNet
from network.gated_cpm_mobilenet import GatedMobilenet
from network.gated_c3d import GatedC3D, GatedStage
from modules.utils import *
# dataset
import configparser
from dataloaders.cmu_hand_data import CMUHand
from tqdm import tqdm
from network.demo_model import GestureNet





if __name__ == "__main__":
    # Logger setup
    mylog.add_log_level("VERBOSE", logging.INFO - 5)
    mylog.add_log_level("MICRO", logging.DEBUG - 5)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Need to set encoding or Windows will choke on ellipsis character in
    # PyTorch tensor formatting
    handler = logging.FileHandler("logs/demo_c3d.log", "w", "utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    root_logger.addHandler(handler)

    pretrained_weights = "ckpt/cpm_r3_model_epoch1100.pth"
    # pretrained_weights = None
    full_net = GestureNet(weights_file=pretrained_weights)
    heatmap_net = full_net.heatmap_net
    net = full_net.c3d_net
    gate_network = net.gate
    ################### pre-trained
    pretrained = False
    if pretrained:
        start = 2
        filename = model_file("ckpt/gated_c3d/", start, ".latest")
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            load_model(net, state_dict, load_gate=True, strict=True)
    else:
        start = 0

    ### GPU support ###
    cuda = torch.cuda.is_available()
    device_ids = [2, 3]

    if cuda:
        heatmap_net = heatmap_net.cuda(device_ids[0])
        net = net.cuda(device_ids[0])
        if len(device_ids) > 1:
            heatmap_net = torch.nn.DataParallel(heatmap_net, device_ids=device_ids)
            net = torch.nn.DataParallel(net, device_ids=device_ids)
            print("Using multi-gpu: ", device_ids)
        else:
            print("Using single gpu: ", device_ids[0])
    ######################### dataset #######################
    # config = configparser.ConfigParser()
    # config.read('conf.text')
    # train_data_dir = config.get('data', 'train_data_dir')
    # train_label_dir = config.get('data', 'train_label_dir')
    batch_size = 16 * len(device_ids)
    # train_data = CMUHand(data_dir=train_data_dir, label_dir=train_label_dir)
    # train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # gesture dataset
    from dataloaders.dataset import VideoDataset
    subset = ['No gesture', 'Swiping Down', 'Swiping Up', 'Swiping Left', 'Swiping Right']
    train_data = VideoDataset(dataset='20bn-jester', split='train', clip_len=16, subset=subset)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    ######################### learner #######################
    # GatePolicyLearner
    import math
    import torch.optim as optim
    import nnsearch.pytorch.gated.learner as glearner
    lambda_gate = 1.0
    learning_rate = 4e-4
    # nclasses = 27
    # complexity_weights = []
    # for (m, in_shape) in net.gated_modules:
    #     complexity_weights.append(1.0) # uniform
    # lambda_gate = lambda_gate * math.log(nclasses)
    optimizer = optim.SGD( net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4 )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=1e-2, eps=1e-9, verbose=True)


    # gate_control = uniform_gate()
    gate_control = constant_gate(1.0)

    gate_loss = glearner.usage_gate_loss( penalty_fn)
    criterion = None
    learner = glearner.GatedDataPathLearner(net, optimizer, learning_rate,
                                            gate_network, gate_control, criterion=criterion, scheduler=scheduler)

    ######################### train #######################
    # start = 0
    train_epochs = 50
    seed = 1
    for epoch in range(start, start + train_epochs):
        print("==== Train: Epoch %s: seed=%s", epoch, seed)
        batch_idx = 0
        nbatches = math.ceil(len(train_data) / batch_size)
        learner.start_train(epoch, seed)
        running_corrects = 0.0
        for i, data in enumerate(tqdm(train_dataset)):
            inputs, labels = data
            if cuda:
                inputs = inputs.cuda(device_ids[0])
                labels = labels.cuda(device_ids[0])
            # generate intermidiate heatmaps
            heatmaps = full_net.get_heatmaps(inputs, torch.tensor(1.0).cuda(device_ids[0]))
            yhat = learner.forward(i, heatmaps, labels)
            loss = learner.backward(i, yhat, labels)
            probs = nn.Softmax(dim=1)(yhat)
            preds = torch.max(probs, 1)[1]
            batch_corrects = torch.sum(preds == labels.data).float()
            running_corrects += batch_corrects
            if i % 10 == 0:
                print("Step [{}] loss: {:.4f}, accuracy: {:.4f}".format(i, loss, batch_corrects / labels.size()[0]))

            batch_idx += 1
        print("Epoch end, training accuracy: {:.4f}".format(running_corrects / len(train_data)))
        learner.finish_train(epoch)
        learner.scheduler_step(loss, epoch)
        checkpoint(net, "ckpt/gated_c3d", epoch + 1, learner)
        # checkpoint(epoch + 1, learner)
        # Save final model if we haven't done so already
    # if args.train_epochs % args.checkpoint_interval != 0:
    #     checkpoint(start + args.train_epochs, learner, force_eval=True)