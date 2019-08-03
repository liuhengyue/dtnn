import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
import nnsearch.logging as mylog
import logging
import nnsearch.pytorch.gated.strategy as strategy
from nnsearch.pytorch.gated.module import GatedChainNetwork
from network.gated_cpm_mobilenet import GatedMobilenet
from network.gated_c3d import GatedC3D, GatedStage
from modules.utils import *
# dataset
import configparser
from dataloaders.cmu_hand_data import CMUHand
from tqdm import tqdm
# order: "kernel_size", "stride", "padding", "nlayers", "nchannels", "ncomponents"

def make_heatmap_net():
    # it should return a dict as the input to GestureNet
    #### first network: frome raw images to keypoint heatmaps
    backbone_stages = [GatedStage("conv", 3, 2, 0, 1, 32, 4), GatedStage("dw_conv", 3, 1, 1, 1, 64, 4),
                       GatedStage("dw_conv", 3, 2, 0, 1, 128, 4), GatedStage("dw_conv", 3, 1, 1, 1, 128, 4),
                       GatedStage("dw_conv", 3, 2, 0, 1, 256, 4), GatedStage("dw_conv", 3, 1, 1, 1, 256, 4),
                       GatedStage("dw_conv", 3, 1, 1, 1, 512, 4), GatedStage("dw_conv", 3, 1, 1, 1, 512, 4),
                       GatedStage("dw_conv", 3, 1, 1, 4, 512, 4), GatedStage("conv", 3, 1, 1, 1, 256, 4),
                       GatedStage("conv", 3, 1, 1, 1, 128, 4)]

    initial_stage = [GatedStage("conv", 3, 1, 1, 3, 128, 4), GatedStage("conv", 1, 1, 0, 1, 512, 4),
                     GatedStage("conv", 1, 1, 0, 1, 21, 1)]

    heatmap_stages = {"backbone": backbone_stages, "initial": initial_stage}
    gate = make_sequentialGate(heatmap_stages)
    in_shape = (3, 368, 368)
    num_classes = 21
    return {"backbone": backbone_stages, "initial": initial_stage, "gate": gate,
            "in_shape":in_shape, "num_classes": num_classes}

def make_c3d_net():
    # it should return a dict as the input to GestureNet
    #### second network: from keypoint heatmaps to gestures
    c3d_stages = [GatedStage("conv", 3, 1, 1, 1, 64, 4), GatedStage("pool", (1, 2, 2), (1, 2, 2), 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 1, 128, 2), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 2, 256, 4), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 2, 512, 4), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 2, 512, 4), GatedStage("pool", 2, 2, 0, 1, 0, 0), ]

    fc_stages = [GatedStage("fc", 0, 0, 0, 2, 1024, 2)]

    stages = {"c3d": c3d_stages, "fc": fc_stages}
    gate = make_sequentialGate(stages)
    in_shape = (21, 16, 45, 45)
    num_classes = 5
    return {"c3d": c3d_stages, "fc": fc_stages, "gate": gate,
            "in_shape":in_shape, "num_classes": num_classes}

class GestureNet():
    """
    The GestureNet network.
    For this network, all the init parameters should be seperate into two groups for the two sub-networks.
    """

    def __init__(self, heatmap_net_pars, c3d_pars, dropout=0.2, **kwargs):


        self.heatmap_net = GatedMobilenet(heatmap_net_pars["gate"], heatmap_net_pars["in_shape"], heatmap_net_pars["num_classes"],
                                          heatmap_net_pars["backbone"], None, heatmap_net_pars["initial"], [], dropout=dropout)

        self.load_pretrained_weights()
        self.heatmap_net.eval()

        self.c3d_net = GatedC3D(c3d_pars["gate"], c3d_pars["in_shape"],
                                          c3d_pars["num_classes"],
                                          c3d_pars["c3d"], c3d_pars["fc"],
                                          dropout=dropout)



    def load_pretrained_weights(self):
        filename = model_file("ckpt/gated_cpm", 100, ".latest")
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            load_model(self.heatmap_net, state_dict,
                       load_gate=True, strict=True)
        # set required grad
        for name, param in self.heatmap_net.named_parameters():
            # print(name)
            param.requires_grad = False


    def get_heatmaps(self, x, u1=None):
        """
        C3D takes input shape: (BatchSize, 3, num_frames, 368, 368)
        Mobile_CPM takes input shape: (BatchSize, 3, 368, 368)
        x shape: (BatchSize, 3, num_frames, 368, 368)
        Should convert (BatchSize, 3, num_frames, 368, 368) ->
                       (BatchSize * num_frames, 3, 368, 368)
        """
        B, C, N, H, W = x.size()
        batch_heatmaps = []
        for b in range(B):
            x_slice = x[b] # (3, N, H, W)
            x_slice = x_slice.permute(1, 0, 2, 3) # (N, 3, H, W)
            heatmap, _ = self.heatmap_net(x_slice, u1) # (N, 3 stages, 21, 45, 45)
            # right now, just one stage so do not slice
            heatmap_final_stage = heatmap
            # heatmap_final_stage = heatmap[:, -1, :, :, :] # (N, 21, 45, 45)
            heatmap_final_stage = heatmap_final_stage.permute(1, 0, 2, 3) # (21, N, 45, 45)
            batch_heatmaps.append(heatmap_final_stage)

        batch_heatmaps = torch.stack(batch_heatmaps)

        return batch_heatmaps





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

    heatmap_net_pars = make_heatmap_net()

    c3d_pars = make_c3d_net()


    full_net = GestureNet(heatmap_net_pars, c3d_pars)
    net = full_net.c3d_net
    # print(net)
    # summary(net, [(3, 32, 32), (1,)])

    # x = torch.rand(2, 3, 16, 64, 64)
    # y, g = net(Variable(x), torch.tensor(0.5), torch.tensor(0.5))
    # print(y)
    # y[0].backward
    ### GPU support ###
    # right now, it can only work on single gpu with number 0, parallel not working
    # gate and inputs are on different gpus
    cuda = torch.cuda.is_available()
    device_ids = [0]

    if cuda:
        if len(device_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=device_ids)
            print("Using multi-gpu: ", device_ids)
        else:
            net = net.cuda(device_ids[0])
            full_net.heatmap_net = full_net.heatmap_net.cuda(device_ids[0])
            print("Using single gpu: ", device_ids[0])
    ######################### dataset #######################
    # config = configparser.ConfigParser()
    # config.read('conf.text')
    # train_data_dir = config.get('data', 'train_data_dir')
    # train_label_dir = config.get('data', 'train_label_dir')
    batch_size = 32
    # train_data = CMUHand(data_dir=train_data_dir, label_dir=train_label_dir)
    # train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # gesture dataset
    from dataloaders.dataset import VideoDataset
    subset = ['No gesture', 'Thumb Down', 'Thumb Up', 'Swiping Left', 'Swiping Right']
    train_data = VideoDataset(dataset='20bn-jester', split='train', clip_len=16, subset=subset)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    ######################### learner #######################
    # GatePolicyLearner
    import math
    import torch.optim as optim
    import nnsearch.pytorch.gated.learner as glearner
    lambda_gate = 1.0
    learning_rate = 0.01
    # nclasses = 27
    # complexity_weights = []
    # for (m, in_shape) in net.gated_modules:
    #     complexity_weights.append(1.0) # uniform
    # lambda_gate = lambda_gate * math.log(nclasses)
    optimizer = optim.SGD( net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4 )
    gate_network = net.gate

    gate_control = uniform_gate()

    gate_loss = glearner.usage_gate_loss( penalty_fn)
    criterion = None
    learner = glearner.GatedDataPathLearner(net, optimizer, learning_rate,
                                            gate_network, gate_control, criterion=criterion)

    ######################### train #######################
    start = 0
    train_epochs = 50
    seed = 1
    for epoch in range(start, start + train_epochs):
        print("==== Train: Epoch %s: seed=%s", epoch, seed)
        batch_idx = 0
        nbatches = math.ceil(len(train_data) / batch_size)
        learner.start_train(epoch, seed)
        for i, data in enumerate(tqdm(train_dataset)):
            inputs, labels = data
            if cuda:
                inputs = inputs.cuda(device_ids[0])
                labels = labels.cuda(device_ids[0])
            # generate intermidiate heatmaps
            heatmaps = full_net.get_heatmaps(inputs, torch.tensor(1.0))
            yhat = learner.forward(i, heatmaps, labels)
            learner.backward(i, yhat, labels)

            batch_idx += 1
        learner.finish_train(epoch)
        checkpoint(net, "ckpt/gated_c3d", epoch + 1, learner)
        # checkpoint(epoch + 1, learner)
        # Save final model if we haven't done so already
    # if args.train_epochs % args.checkpoint_interval != 0:
    #     checkpoint(start + args.train_epochs, learner, force_eval=True)