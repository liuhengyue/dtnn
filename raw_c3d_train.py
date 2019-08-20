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
# from dataloaders.cmu_hand_data import CMUHand
from tqdm import tqdm
from network.demo_model import GestureNet

def c3d():
    span_factor = 2

    c3d_stages = [GatedStage("conv", 3, 1, 1, 1, 64, 1), GatedStage("pool", (1, 2, 2), (1, 2, 2), 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 1, 128 * span_factor, 8), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 2, 256 * span_factor, 16), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 2, 512 * span_factor, 16), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 2, 512, 16), GatedStage("pool", 2, 2, 0, 1, 0, 0), ]

    fc_stages = [GatedStage("fc", 0, 0, 0, 2, 512, 4)]

    # non gated
    # c3d_stages = [GatedStage("conv", 3, 1, 1, 1, 64, 1), GatedStage("pool", (1, 2, 2), (1, 2, 2), 0, 1, 0, 0),
    #               GatedStage("conv", 3, 1, 1, 1, 128, 1), GatedStage("pool", 2, 2, 0, 1, 0, 0),
    #               GatedStage("conv", 3, 1, 1, 2, 256, 1), GatedStage("pool", 2, 2, 0, 1, 0, 0),
    #               GatedStage("conv", 3, 1, 1, 2, 512, 1), GatedStage("pool", 2, 2, 0, 1, 0, 0),
    #               GatedStage("conv", 3, 1, 1, 2, 512, 1), GatedStage("pool", 2, 2, 0, 1, 0, 0), ]
    #
    # fc_stages = [GatedStage("fc", 0, 0, 0, 2, 512, 1)]

    stages = {"c3d": c3d_stages, "fc": fc_stages}
    gate = make_sequentialGate(stages)
    # in_shape = (21, 16, 45, 45)
    in_shape = (3, 16, 368, 368) # for raw input
    num_classes = 5
    c3d_pars = {"c3d": c3d_stages, "fc": fc_stages, "gate": gate,
            "in_shape": in_shape, "num_classes": num_classes}

    c3d_net = GatedC3D(c3d_pars["gate"], c3d_pars["in_shape"],
                       c3d_pars["num_classes"], c3d_pars["c3d"], c3d_pars["fc"], dropout=0)

    return c3d_net



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

    net = c3d()
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
    # cuda = False
    device_ids = [0, 1, 2, 3]

    if cuda:
        net = net.cuda(device_ids[0])
        if len(device_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=device_ids)
            print("Using multi-gpu: ", device_ids)
        else:
            print("Using single gpu: ", device_ids[0])
    ######################### dataset #######################
    # config = configparser.ConfigParser()
    # config.read('conf.text')
    # train_data_dir = config.get('data', 'train_data_dir')
    # train_label_dir = config.get('data', 'train_label_dir')
    batch_size = 2 * len(device_ids)
    # train_data = CMUHand(data_dir=train_data_dir, label_dir=train_label_dir)
    # train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # gesture dataset
    from dataloaders.dataset import VideoDataset
    subset = ['No gesture', 'Swiping Down', 'Swiping Up', 'Swiping Left', 'Swiping Right']
    train_data = VideoDataset(dataset='20bn-jester', split='train', clip_len=16, subset=subset)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    ######################### learner #######################
    # GatePolicyLearner
    import math
    import torch.optim as optim
    import nnsearch.pytorch.gated.learner as glearner
    lambda_gate = 1.0
    learning_rate = 4e-3
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
            yhat = learner.forward(i, inputs, labels)
            loss = learner.backward(i, yhat, labels)
            probs = nn.Softmax(dim=1)(yhat)
            preds = torch.max(probs, 1)[1]
            batch_corrects = torch.sum(preds == labels.data).float()
            running_corrects += batch_corrects
            if i % 10 == 0:
                print("Step [{}] loss: {:.4f}, accuracy: {:.4f}".format(i, loss, batch_corrects / labels.size()[0]))

            batch_idx += 1
            # if i == 11:
            #     break

        print("Epoch end, training accuracy: {:.4f}".format(running_corrects / len(train_data)))
        learner.finish_train(epoch)
        learner.scheduler_step(loss, epoch)
        checkpoint(net, "ckpt/gated_raw_c3d", epoch + 1, learner)
        # checkpoint(epoch + 1, learner)
        # Save final model if we haven't done so already
    # if args.train_epochs % args.checkpoint_interval != 0:
    #     checkpoint(start + args.train_epochs, learner, force_eval=True)