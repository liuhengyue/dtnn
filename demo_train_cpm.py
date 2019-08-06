import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
import nnsearch.logging as mylog
import logging
from network.gated_cpm_mobilenet import GatedMobilenet, GatedStage
# dataset
import configparser
from dataloaders.cmu_hand_data import CMUHand

from modules.utils import *



if __name__ == "__main__":
    # Logger setup
    mylog.add_log_level("VERBOSE", logging.INFO - 5)
    mylog.add_log_level("MICRO", logging.DEBUG - 5)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Need to set encoding or Windows will choke on ellipsis character in
    # PyTorch tensor formatting
    handler = logging.FileHandler("logs/demo.log", "w", "utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    root_logger.addHandler(handler)

    # order: "kernel_size", "stride", "padding", "nlayers", "nchannels", "ncomponents"
    backbone_stages = [GatedStage("conv", 3, 2, 0, 1, 32, 4), GatedStage("dw_conv", 3, 1, 1, 1, 64, 4),
                       GatedStage("dw_conv", 3, 2, 0, 1, 128, 4), GatedStage("dw_conv", 3, 1, 1, 1, 128, 4),
                       GatedStage("dw_conv", 3, 2, 0, 1, 256, 4), GatedStage("dw_conv", 3, 1, 1, 1, 256, 4),
                       GatedStage("dw_conv", 3, 1, 1, 1, 512, 4), GatedStage("dw_conv", 3, 1, 1, 1, 512, 4),
                       GatedStage("dw_conv", 3, 1, 1, 4, 512, 4), GatedStage("conv", 3, 1, 1, 1, 256, 4),
                       GatedStage("conv", 3, 1, 1, 1, 128, 4)]

    initial_stage = [GatedStage("conv", 3, 1, 1, 3, 128, 4), GatedStage("conv", 1, 1, 0, 1, 512, 4),
                     GatedStage("conv", 1, 1, 0, 1, 21, 1)]

    # fc_stage = GatedVggStage(1, 512, 2)
    full_stage = {"backbone_stages": backbone_stages, "initial": initial_stage}
    gate = make_sequentialGate(full_stage)


    net = GatedMobilenet(gate, (3, 368, 368), 21, backbone_stages, None, initial_stage, [])
    gate_network = net.gate
    # print(net)
    # summary(net, [(3, 32, 32), (1,)])

    # x = torch.rand(1, 3, 32, 32)
    # print(x.size())
    # y = net(Variable(x), torch.tensor(0.5))
    # print(y)
    # y[0].backward
    ### GPU support ###
    # right now, it can only work on single gpu with number 0, parallel not working
    # gate and inputs are on different gpus
    cuda = torch.cuda.is_available()
    cuda = False
    device_ids = [0]

    if cuda:
        if len(device_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=device_ids)
        else:
            net = net.cuda(device_ids[0])

    ######################### dataset #######################
    config = configparser.ConfigParser()
    config.read('conf.text')
    train_data_dir = config.get('data', 'train_data_dir')
    train_label_dir = config.get('data', 'train_label_dir')
    batch_size = 2
    train_data = CMUHand(data_dir=train_data_dir, label_dir=train_label_dir)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    ######################### learner #######################
    # GatePolicyLearner
    import math
    import torch.optim as optim
    import nnsearch.pytorch.gated.learner as glearner
    # lambda_gate = 1.0
    learning_rate = config.getfloat('training', 'learning_rate')
    nclasses = 21
    # complexity_weights = []
    # for (m, in_shape) in net.gated_modules:
    #     complexity_weights.append(1.0) # uniform
    # lambda_gate = lambda_gate * math.log(nclasses)
    # optimizer = optim.SGD( net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4 )
    # cpm default optim
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    gate_control = uniform_gate()

    gate_loss = glearner.usage_gate_loss( penalty_fn)
    criterion = torch.nn.MSELoss(reduction='mean')
    learner = glearner.GatedDataPathLearner(net, optimizer, learning_rate,
                                            gate_network, gate_control, criterion=criterion)


    ######################### train #######################

    start = 0
    train_epochs = 100
    seed = 1
    n_refine_stages = 1
    for epoch in range(start, start + train_epochs):
        print("==== Train: Epoch %s: seed=%s", epoch, seed)
        batch_idx = 0
        nbatches = math.ceil(len(train_data) / batch_size)
        learner.start_train(epoch, seed)
        for i, data in enumerate(train_dataset):
            inputs, labels, _, _ = data
            labels = torch.stack([labels] * (n_refine_stages + 1), dim=1)
            if cuda:
                inputs = inputs.cuda(device_ids[0])
                labels = labels.cuda(device_ids[0])
            yhat = learner.forward(i, inputs, labels)
            learner.backward(i, yhat, labels)

            batch_idx += 1
        learner.finish_train(epoch)
        checkpoint(net, "ckpt/gated_cpm", epoch + 1, learner)
        # checkpoint(epoch + 1, learner)
        # Save final model if we haven't done so already
    # if args.train_epochs % args.checkpoint_interval != 0:
    #     checkpoint(start + args.train_epochs, learner, force_eval=True)