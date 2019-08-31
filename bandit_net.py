import torch.nn as nn
import torch
import shape_flop_util as util

class ContextualBanditNet(nn.Module):
    '''
    Right now, it should takes in the intermeidate output from
    the datanetwork, then go through a small network for contextual-aware
    purpose to generate states.
    '''
    def __init__(self, context_network=None):
        super(ContextualBanditNet, self).__init__()

        self.fc_size = 800

        self.pgconv = nn.Sequential(
            nn.Conv3d(3, 64, (3,3,3), 1, 0),
            nn.Conv3d(64, 32, (3,3,3), 1, 0),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3,3,3), 1, 0),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Conv3d(32, 16, (3,3,3), 1, 1),
            nn.ReLU(),
            nn.Conv3d(16, 16, (3, 3, 3), 1, 0),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Conv3d(16, 4, (3, 3, 3), 1, 0),
            nn.ReLU(),
            nn.Conv3d(4, 4, (3, 3, 3), 1, 0),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Conv3d(4, 1, (3, 3, 3), 1, 0),
            nn.ReLU(),
            # nn.Conv3d(4, 4, (3, 3, 3), 1, 0),
            # nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2))
            )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_size, 64),
            nn.ReLU(),
            nn.Linear(64, 15),
        )
        self.sm = nn.Sigmoid()

    # self.lstm = nn.RNN(self.input_dim , self.hidden_dim, self.layer_dim, batch_first=True)
    # self.features = context_network
    # self.q = nn.Sequential(
    # nn.Linear(64, 10),
    # nn.Softmax())
    def forward(self, x):

        x = self.pgconv(x)
        print(x.size())

        x = x.view(-1, self.fc_size)
        # print(x.size())
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

if __name__ == "__main__":
    input_shape = (3, 16, 368, 368)
    input = torch.randn(input_shape)
    net = ContextualBanditNet().cuda()
    # print(net)
    from torchsummary import summary
    summary(net, input_shape, device="cuda")