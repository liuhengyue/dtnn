import torch.nn as nn
import torch
import shape_flop_util as util

class ContextualBanditNet(nn.Module):
    def __init__(self, context_network=None):
        super(ContextualBanditNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 5, 5, 1)
        # self.conv2 = nn.Conv2d(5, 10, 5, 1)
        # self.hidden_dim = 100
        # self.layer_dim = 1
        # self.input_dim  = 160
        self.features = nn.Sequential(
            nn.Conv3d(21, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Conv3d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)))
        self.pgconv = nn.Sequential(
            nn.Conv3d(128, 64, 3, 2, 0),
            nn.ReLU(),
            nn.Conv3d(64, 16, 1, 1, 1),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(2304, 526),
            nn.Sigmoid(),
            nn.Linear(526, 248),
            nn.ReLU(),
            nn.Linear(248, 15),
        )
        self.sm = nn.Sigmoid()

    # self.lstm = nn.RNN(self.input_dim , self.hidden_dim, self.layer_dim, batch_first=True)
    # self.features = context_network
    # self.q = nn.Sequential(
    # nn.Linear(64, 10),
    # nn.Softmax())
    def forward(self, x):
        #         x = F.relu(self.conv1(x))
        #         x = F.max_pool2d(x, 2, 2)
        #         x = F.relu(self.conv2(x))
        #         x = F.max_pool2d(x, 2, 2)
        # print("SHAPE", x)
        x = self.features(x)
        x = self.pgconv(x)
        x = x.view(-1)
        x = self.fc(x)
        #print("BEFORE SOFTMAX:, ", x)
        x = self.sm(x)
        #print("AFTER SOFTMAX:, ", x)

        # print("SHAPE BEFORE FLATTEN", x.shape)
        # x = x[0].view(-1)
        # print(x.shape)
        #         h0 = autograd.Variable(torch.randn(self.layer_dim, x.size(0), self.hidden_dim))
        #         x, hidden = self.lstm(x, h0)
        return x  # , hidden


@util.flops.register(ContextualBanditNet)
def _(layer, in_shape):
    # f1 = flops( layer.features, in_shape )
    # f2 = flops( layer.q, (1, 64) )
    return util.Flops(10000)
    # FIXME: This is wrong if we add fields to Flops tuple
    return util.Flops(f1.macc + f2.macc)