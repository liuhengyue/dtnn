import torch.nn as nn
import torch
import shape_flop_util as util
from nnsearch.pytorch.torchx import *
import numpy as np

class ContextualBanditNet(nn.Module):
    '''
    Right now, it should takes in the intermeidate output from
    the datanetwork, then go through a small network for contextual-aware
    purpose to generate states.
    '''
    def __init__(self, context_network=None):
        super(ContextualBanditNet, self).__init__()

        self.ngate_levels = 10
        inc = 1.0 / self.ngate_levels
        # u = 0 does not make sense
        # if 10 levels: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
        self._us = torch.tensor([i * inc for i in range(1, self.ngate_levels + 1)], requires_grad=False)

        self.fc_size = 360

        self.pgconv = nn.Sequential(
            nn.Conv3d(3, 32, (3,7,7), 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3,7,7), 2, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3, 7, 7), 1, 0),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3,3,3), 2, 1),
            nn.ReLU(),
            nn.Conv3d(32, 16, (3, 3, 3), (1, 2, 2), 1),
            nn.ReLU(),
            nn.Conv3d(16, 16, (3, 3, 3), (1, 2, 2), 1),
            nn.ReLU(),
            nn.Conv3d(16, 8, (3, 3, 3), (1, 2, 2), 1),
            nn.ReLU(),
            )

        # self.pgconv = nn.Sequential(
        #     nn.Conv3d(3, 64, (3, 3, 3), 1, 1),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 128, (3, 3, 3), 2, 1),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 256, (3, 3, 3), 1, 0),
        #     nn.ReLU(),
        #     nn.Conv3d(256, 128, (3, 3, 3), 2, 1),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 64, (3, 3, 3), (1, 2, 2), 1),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 32, (3, 3, 3), (1, 2, 2), 1),
        #     nn.ReLU(),
        #     nn.Conv3d(32, 1, (3, 3, 3), (1, 2, 2), 1),
        #     nn.ReLU(),
        # )


        self.fc = nn.Sequential(
            nn.Linear(self.fc_size, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU()
        )
        # self.sm = nn.Sigmoid()

    def forward(self, x):

        x = self.pgconv(x)
        # print(x.size())

        x = x.view(-1, self.fc_size)
        x = self.fc(x)
        # x = self.sm(x)

        # x = x.view(-1, 5, 10)
        #print("AFTER SOFTMAX:, ", x)


        return x  # , hidden


@util.flops.register(ContextualBanditNet)
def _(net, in_shape):
    total_macc = 0.0
    for m in net.pgconv:
        if isinstance(m, nn.Conv3d):
            total_macc += util.flops(m, in_shape).macc
        in_shape = output_shape(m, in_shape)

    for m in net.fc:
        if isinstance(m, nn.Linear):
            total_macc += util.flops(m, in_shape).macc
        in_shape = output_shape(m, in_shape)
    return util.Flops(total_macc)


class ManualController():
    def __init__(self):
        self.history = np.array([0.5])
        self.discount = 0.9

    def add_to_history(self, pred):
        # add a prediction to controller's history
        # 'No Gesture' - 0 everytime it receives a 'No Gesture' class, it add
        # a zero to the history; else add a one
        score = np.array([0.0]) if pred == "No gesture" else np.array([1.0])
        self.history = self.history * self.discount
        self.history = np.concatenate((self.history, score))

        # reset if too much history
        if len(self.history) > 100:
            self.reset()

    def get_utilization(self):
        # return a high value if prediction has changed
        if self.history[-1] == 1.0:
            return torch.tensor([1.0])
        else:
            return torch.tensor([sum(self.history) / len(self.history)])

    def reset(self):
        self.history = np.array([0.5])



if __name__ == "__main__":
    input_shape = (3, 16, 100, 160)
    input = torch.randn(input_shape)
    net = ContextualBanditNet().cuda()
    net.eval()
    # print(net)
    from torchsummary import summary
    summary(net, input_shape, device="cuda")
    print(util.flops(net, (3, 16, 100, 160)))