import torch
import torch.nn as nn
import shape_flop_util as util

class ContextualQNetwork(nn.Module):
    def __init__(self, observation_space, action_space,
                 context_network, ctx_out_channels, nhidden):
        super().__init__()
        self.sysdim = observation_space.spaces[0].shape[0]
        self.ctx_out_channels = ctx_out_channels
        # FIXME: Hardcoded action space from 'solar'
        nactions = action_space.n
        print("NUMBER OF ACTIONS=", nactions)
        self.features = context_network
        self.q = nn.Sequential(
            nn.Linear(self.sysdim + self.ctx_out_channels, nhidden, bias=False),
            nn.ReLU(),
            nn.Linear(nhidden, nactions, bias=False))

    def forward(self, s):
        xsys, ximg = s
        f = self.features(ximg)
        x = torch.cat([f, xsys], dim=1)
        print("Q VALUE OUTPUTTED", self.q(s))
        return self.q(x)

@util.flops.register(ContextualQNetwork)
def _(layer, in_shape):
    f1 = flops(layer.features, in_shape)
    f2 = flops(layer.q, [layer.sysdim + layer.ctx_out_channels])
    # FIXME: This is wrong if we add fields to Flops tuple
    return Flops(f1.macc + f2.macc)