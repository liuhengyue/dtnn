from collections import namedtuple
from functools import reduce
import logging
import operator

import torch
import torch.nn as nn

from nnsearch.pytorch.gated.module import (BlockGatedConv2d, BlockGatedFullyConnected, 
    GatedChainNetwork, GatedModule)
from modules.conv import conv, gatedConvBlock, gatedDwConvBlock, gated3dConvBlock
from network.cpm_mobilenet import *
import nnsearch.pytorch.gated.strategy as strategy
from nnsearch.pytorch.modules import FullyConnected
from modules.utils import *
import nnsearch.pytorch.torchx as torchx
from torchsummary import summary
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------

GatedStage = namedtuple("GatedStage",
                           ["name", "kernel_size", "stride", "padding", "nlayers", "nchannels", "ncomponents"])



# ----------------------------------------------------------------------------
class GatedRefinementStageBlock(GatedModule, nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    # def forward(self, x):
    #     initial_features = self.initial(x)
    #     trunk_features = self.trunk(initial_features)
    #     return initial_features + trunk_features

    def forward(self, x, g):
        initial_features = self.initial(x)
        gi = g[:, 0].contiguous()
        gu = torchx.unsqueeze_right_as(gi, initial_features)
        initial_features = gu * initial_features

        trunk_features = self.trunk(initial_features)
        gi = g[:, 1].contiguous()
        gu = torchx.unsqueeze_right_as(gi, trunk_features)
        trunk_features = gu * trunk_features

        return initial_features + trunk_features


    @property
    def ncomponents(self):
        return len(self.components)


    @property
    def vectorize(self):
        return self._vectorize


    @vectorize.setter
    def vectorize(self, value):
        self._vectorize = value


class GatedMobilenet(nn.Module):

    def __init__(self, gate, in_shape, nclasses, bb_stages, fc_stage,
                 initial_stage, refine_stage, batchnorm=False, dropout=0.5,
                 normalize=True, reverse_normalize=False, gbar_info=False,
                 n_refine_stages=3, **kwargs):
        super().__init__()
        # self.gate = gate
        self.bb_stages = bb_stages
        self.fc_stage = fc_stage
        self.initial_stage = initial_stage
        self.refine_stage = refine_stage
        self.nclasses = nclasses
        self.n_refine_stages = n_refine_stages
        self.batchnorm = batchnorm # batch norm already defined in modules/conv.py
        self.dropout = dropout
        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.backbone = nn.ModuleList()
        self.initial_stage = None
        self.refinement_stages = nn.ModuleList()
        # self.modules = []
        self.tmp_gated_modules = [] # GatedChainNetwork has a property called gated_modules
        
        self.__set_backbone()
        # add initial and refinement stages here
        self.__set_initial_stage()

        self.__set_refinement_stage()
        # refine_stage = RefinementStageBlock(self.in_channels, self.in_channels)
        # self.modules.append(refine_stage)
        # self.tmp_gated_modules.append((refine_stage, (self.in_channels,) + self.in_shape[1:]))
        # self.__set_fc()
        # self.__set_classification_layer()
        # self.fn = nn.ModuleList( modules )
        # print("modules------------------",modules)
        # print("gated modules------------------", gated_modules)
        self.gate = gate
        self.backbone = nn.ModuleList(self.backbone)
        self._gated_modules = self.tmp_gated_modules
        self.normalize = normalize
        self.reverse_normalize = reverse_normalize
        self._gbar_info = gbar_info

    @property
    def gated_modules(self):
        yield from self._gated_modules

    def _normalize(self, g):
        z = 1e-12 + torch.sum(g, dim=1, keepdim=True)  # Avoid divide-by-zero
        if self.reverse_normalize:
            return g * g.size(1) / z
        else:
            return g / z

    def _log_gbar(self, gbar):
        log.debug("network.gbar:\n%s", gbar)
        if self._gbar_info:
            N = torch.sum((gbar.data > 0).float(), dim=1, keepdim=True)
            # log.info( "network.log_gbar.N:\n%s", N )
            g = N * gbar.data
            # log.info( "network.log_gbar.g:\n%s", g )
            h = torchx.histogram(g,
                                 bins=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
            log.info("network.g.hist:\n%s", torchx.pretty_histogram(h))

    def set_gate_control(self, u):
        self._u = u

    def forward(self, x, u=None):
        """
        Returns:
          `(y, gs)` where:
            `y`  : Network output
            `gs` : `[(g, info)]` List of things returned from `self.gate` in same
              order as `self.gated_modules`. `g` is the actual gate matrix, and
              `info` is any additional things returned (or `None`).
        """

        def expand(gout):
            if isinstance(gout, tuple):
                g, info = gout  # Fail fast on unexpected extra outputs
                return (g, info)
            else:
                return (gout, None)

        gs = []
        # FIXME: This is a hack for FasterRCNN integration. Find a better way.
        if u is None:
            self.gate.set_control(self._u)
        else:
            self.gate.set_control(u)
        # TODO: With the current architecture, set_control() has to happen before
        # reset() in case reset() needs to run the gate network. Should `u` be a
        # second parameter to reset()? Should it be an argument to gate() as well?
        # How do we support gate networks that require the outputs of arbitrary
        # layers in the data network in a modular way?
        self.gate.reset(x)
        for m in self.backbone:
            # print(type(m))
            if isinstance(m, GatedModule):
                self.gate.next_module(m)
                g, info = expand(self.gate(x))
                gs.append((g, info))
                if self.normalize:
                    g = self._normalize(g)
                self._log_gbar(g)
                # print(m, g)
                x = m(x, g)
            else:
                x = m(x)
            # print(x.size())

        stages_output = self.initial_stage(x)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([x, stages_output[-1]], dim=1)))
        y = torch.stack(stages_output, dim=1)

        log.debug("network.x: %s", x)
        return y, gs

    def __set_backbone(self):
        for i, stage in enumerate(self.bb_stages):
            for _ in range(stage.nlayers):
                if stage.ncomponents > 1:
                    # first conv stage is just a single conv block
                    if stage.name == "dw_conv":
                        m, gated_m, in_shape = gatedDwConvBlock(stage.ncomponents, self.in_shape, self.in_channels, stage.nchannels, 
                            kernel_size=stage.kernel_size, stride=stage.stride, padding=stage.padding)
                    elif stage.name == "conv":
                        m, gated_m, in_shape = gatedConvBlock(stage.ncomponents, self.in_shape, self.in_channels, stage.nchannels,
                            kernel_size=stage.kernel_size, stride=stage.stride, padding=stage.padding, bias=True)
                    else:
                        raise Exception("un-recognized stage.name = {}".format(stage.name))
                    self.in_shape = in_shape
                    self.backbone.extend(m)
                    self.tmp_gated_modules.extend(gated_m)
                else:
                    self.backbone.append(nn.Conv2d(
                        self.in_channels, stage.nchannels, stage.kernel_size,
                        stride=stage.stride, padding=stage.padding))
                if self.batchnorm:
                    self.backbone.append(nn.BatchNorm2d(stage.nchannels))
                # modules.append(nn.ReLU())
                self.in_channels = stage.nchannels

    def __set_initial_stage(self, num_channels=128):
        # for i, stage in enumerate(self.initial_stage):
        #     for _ in range(stage.nlayers):
        #         if stage.ncomponents > 1:
        #
        #             m, gated_m, in_shape = gatedConvBlock(stage.ncomponents, self.in_shape, self.in_channels, stage.nchannels,
        #                                     kernel_size=stage.kernel_size, stride=stage.stride, padding=stage.padding, bias=True)
        #             self.in_shape = in_shape
        #             self.modules.extend(m)
        #             self.tmp_gated_modules.extend(gated_m)
        #         else:
        #             self.modules.append(nn.Conv2d(
        #                 self.in_channels, stage.nchannels, stage.kernel_size, padding=stage.padding))
        #         if self.batchnorm:
        #             self.modules.append(nn.BatchNorm2d(stage.nchannels))
        #         # modules.append(nn.ReLU())
        #         self.in_channels = stage.nchannels
        initial_stage = InitialStage(num_channels, self.nclasses)
        self.initial_stage = initial_stage

    def __set_refinement_stage(self, num_channels=128):
        for _ in range(self.n_refine_stages):
            stage = RefinementStage(num_channels + self.nclasses, num_channels, self.nclasses,
                            False)
            self.refinement_stages.append(stage)

    # def forward(self, x, u=None):
    #     x_output, g = super().forward(x, u)
    #     return x_output



# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from torch.autograd import Variable

    import nnsearch.logging as mylog

    # Logger setup
    mylog.add_log_level("VERBOSE", logging.INFO - 5)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Need to set encoding or Windows will choke on ellipsis character in
    # PyTorch tensor formatting
    handler = logging.FileHandler("vgg.log", "w", "utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    root_logger.addHandler(handler)

    # order: "name", "kernel_size", "stride", "padding", "nlayers", "nchannels", "ncomponents"
    backbone_stages = [GatedStage("conv", 3, 2, 0, 1,  32, 1), GatedStage("dw_conv", 3, 1, 1, 1,  64, 2),
                       GatedStage("dw_conv", 3, 2, 0, 1, 128, 2), GatedStage("dw_conv", 3, 1, 1, 1, 128, 2),
                       GatedStage("dw_conv", 3, 2, 0, 1, 256, 2), GatedStage("dw_conv", 3, 1, 1, 1, 256, 2),
                       GatedStage("dw_conv", 3, 1, 1, 1, 512, 2), GatedStage("dw_conv", 3, 1, 1, 1, 512, 2),
                       GatedStage("dw_conv", 3, 1, 1, 4, 512, 2), GatedStage("conv", 3, 1, 1, 1,  256, 2),
                       GatedStage("conv", 3, 1, 1, 1,  128, 2)]

    initial_stage = [GatedStage("conv", 3, 1, 1, 3, 128, 2), GatedStage("conv", 1, 1, 0, 1, 512, 2),
                     GatedStage("conv", 1, 1, 0, 1, 21, 1)]
    # TODO: not implemented yet
    refine_stage = [1]
    

    # fc_stage = GatedStage("fc", 0, 0, 0, 1, 512, 2)
    full_stage = {"backbone_stages": backbone_stages, "initial": initial_stage, "refinement": refine_stage}


    gate = make_sequentialGate(full_stage)



    # groups = []
    # gi = 0
    # for conv_stage in conv_stages:
    #   count = strategy.PlusOneCount( strategy.UniformCount( conv_stage.ncomponents - 1 ) )
    #   gate_modules.append( strategy.NestedCountGate( conv_stage.ncomponents, count ) )
    #   groups.extend( [gi] * `.nlayers )
    #   gi += 1
    # count = strategy.PlusOneCount( strategy.UniformCount( fc_stage.ncomponents - 1 ) )
    # gate_modules.append( strategy.NestedCountGate( fc_stage.ncomponents, count ) )
    # groups.extend( [gi] * fc_stage.nlayers )
    # gate = strategy.GroupedGate( gate_modules, groups )

    net = GatedMobilenet(gate, (3, 368, 368), 21, backbone_stages, None, initial_stage, [])
    net.eval()
    # net = net.cuda()
    # print(net)
    # summary(net, [(3, 368, 368), (1,)])
    x = torch.rand( 1, 3, 368, 368)
    u = torch.tensor(0.5)
    y = net(x, u)
    # print(y.size())
    print( y[0].size() )
    y[0].backward
