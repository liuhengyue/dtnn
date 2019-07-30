import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
import nnsearch.logging as mylog
import logging
import nnsearch.pytorch.gated.strategy as strategy
from network.gated_cpm_mobilenet import GatedMobilenet, GatedStage, GatedVggStage
# dataset
import configparser
from dataloaders.cmu_hand_data import CMUHand

def make_sequentialGate(backbone_stages, initial_stage):
    gate_modules = []

    for conv_stage in backbone_stages:
        for _ in range(conv_stage.nlayers):
            # each stage uses depthwise conv2d with two gated layers except for the first conv stage
            if conv_stage.name == "dw_conv":
                count = strategy.PlusOneCount(strategy.UniformCount(conv_stage.ncomponents - 1))
                gate_modules.append(strategy.NestedCountGate(conv_stage.ncomponents, count))
            count = strategy.PlusOneCount(strategy.UniformCount(conv_stage.ncomponents - 1))
            gate_modules.append(strategy.NestedCountGate(conv_stage.ncomponents, count))

    for conv_stage in initial_stage:
        for _ in range(conv_stage.nlayers):
            count = strategy.PlusOneCount(strategy.UniformCount(conv_stage.ncomponents - 1))
            gate_modules.append(strategy.NestedCountGate(conv_stage.ncomponents, count))

    # for _ in range(fc_stage.nlayers):
    #     count = strategy.PlusOneCount(strategy.UniformCount(fc_stage.ncomponents - 1))
    #     gate_modules.append(strategy.NestedCountGate(fc_stage.ncomponents, count))

    return strategy.SequentialGate(gate_modules)

if __name__ == "__main__":
    # Logger setup
    mylog.add_log_level("VERBOSE", logging.INFO - 5)
    mylog.add_log_level("MICRO", logging.DEBUG - 5)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
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

    gate = make_sequentialGate(backbone_stages, initial_stage)


    net = GatedMobilenet(gate, (3, 368, 368), 21, backbone_stages, None, initial_stage, [])
    # print(net)
    # summary(net, [(3, 32, 32), (1,)])

    # x = torch.rand(1, 3, 32, 32)
    # print(x.size())
    # y = net(Variable(x), torch.tensor(0.5))
    # print(y)
    # y[0].backward

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
    lambda_gate = 1.0
    learning_rate = 0.01
    nclasses = 21
    complexity_weights = []
    for (m, in_shape) in net.gated_modules:
        complexity_weights.append(1.0) # uniform
    lambda_gate = lambda_gate * math.log(nclasses)
    optimizer = optim.SGD( net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4 )
    gate_network = net.gate
    def uniform_gate():
        def f(inputs, labels):
            # return Variable( torch.rand(inputs.size(0), 1).type_as(inputs) )
            umin = 0
            r = 1.0 - umin
            return Variable(umin + r * torch.rand(inputs.size(0)).type_as(inputs))

        return f
    gate_control = uniform_gate()
    def penalty_fn( G, u ):
      return (1 - u) * G
    gate_loss = glearner.usage_gate_loss( penalty_fn)
    criterion = torch.nn.MSELoss(reduction='mean')
    learner = glearner.GatedDataPathLearner(net, optimizer, learning_rate,
                                            gate_network, gate_control, criterion=criterion)

    ######################### train #######################
    start = 0
    train_epochs = 1
    seed = 1
    for epoch in range(start, start + train_epochs):
        print("==== Train: Epoch %s: seed=%s", epoch, seed)
        batch_idx = 0
        nbatches = math.ceil(len(train_data) / batch_size)
        learner.start_train(epoch, seed)
        for i, data in enumerate(train_dataset):
            inputs, labels, _, _ = data

            yhat = learner.forward(i, inputs, labels)
            learner.backward(i, yhat, labels)

            batch_idx += 1
        learner.finish_train(epoch)
        # checkpoint(epoch + 1, learner)
        # Save final model if we haven't done so already
    # if args.train_epochs % args.checkpoint_interval != 0:
    #     checkpoint(start + args.train_epochs, learner, force_eval=True)