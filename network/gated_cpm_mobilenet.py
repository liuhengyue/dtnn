from collections import namedtuple
from functools import reduce
import logging
import operator

import torch
import torch.nn as nn

from nnsearch.pytorch.gated.module import (BlockGatedConv2d, BlockGatedFullyConnected, 
    GatedChainNetwork, GatedModule)
from modules.conv import gatedConvBlock, gatedDwConvBlock, gated3dConvBlock
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

class GatedMobilenet(GatedChainNetwork):

    def __init__(self, gate, in_shape, nclasses, bb_stages, fc_stage,
                 initial_stage, refine_stage, batchnorm=False, dropout=0.5, **kwargs):
        # super().__init__()
        # self.gate = gate
        self.bb_stages = bb_stages
        self.fc_stage = fc_stage
        self.initial_stage = initial_stage
        self.refine_stage = refine_stage
        self.nclasses = nclasses
        self.batchnorm = batchnorm # batch norm already defined in modules/conv.py
        self.dropout = dropout
        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.modules = []
        self.tmp_gated_modules = [] # GatedChainNetwork has a property called gated_modules
        
        self.__set_backbone()
        # add initial and refinement stages here
        self.__set_initial_stage()
        # self.__set_fc()
        # self.__set_classification_layer()
        # self.fn = nn.ModuleList( modules )
        # print("modules------------------",modules)
        # print("gated modules------------------", gated_modules)
        super().__init__(gate, self.modules, self.tmp_gated_modules, **kwargs)

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
                    self.modules.extend(m)
                    self.tmp_gated_modules.extend(gated_m)
                else:
                    self.modules.append(nn.Conv2d(
                        self.in_channels, stage.nchannels, stage.kernel_size, padding=1))
                if self.batchnorm:
                    self.modules.append(nn.BatchNorm2d(stage.nchannels))
                # modules.append(nn.ReLU())
                self.in_channels = stage.nchannels

    def __set_initial_stage(self):
        for i, stage in enumerate(self.initial_stage):
            for _ in range(stage.nlayers):
                if stage.ncomponents > 1:

                    m, gated_m, in_shape = gatedConvBlock(stage.ncomponents, self.in_shape, self.in_channels, stage.nchannels,
                                            kernel_size=stage.kernel_size, stride=stage.stride, padding=stage.padding, bias=True)
                    self.in_shape = in_shape
                    self.modules.extend(m)
                    self.tmp_gated_modules.extend(gated_m)
                else:
                    self.modules.append(nn.Conv2d(
                        self.in_channels, stage.nchannels, stage.kernel_size, padding=stage.padding))
                if self.batchnorm:
                    self.modules.append(nn.BatchNorm2d(stage.nchannels))
                # modules.append(nn.ReLU())
                self.in_channels = stage.nchannels



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
    backbone_stages = [GatedStage("conv", 3, 2, 0, 1,  32, 4), GatedStage("dw_conv", 3, 1, 1, 1,  64, 4),
                       GatedStage("dw_conv", 3, 2, 0, 1, 128, 4), GatedStage("dw_conv", 3, 1, 1, 1, 128, 4),
                       GatedStage("dw_conv", 3, 2, 0, 1, 256, 4), GatedStage("dw_conv", 3, 1, 1, 1, 256, 4),
                       GatedStage("dw_conv", 3, 1, 1, 1, 512, 4), GatedStage("dw_conv", 3, 1, 1, 1, 512, 4),
                       GatedStage("dw_conv", 3, 1, 1, 4, 512, 4), GatedStage("conv", 3, 1, 1, 1,  256, 4),
                       GatedStage("conv", 3, 1, 1, 1,  128, 4)]

    initial_stage = [GatedStage("conv", 3, 1, 1, 3, 128, 4), GatedStage("conv", 1, 1, 0, 1, 512, 4),
                     GatedStage("conv", 1, 1, 0, 1, 21, 1)]
    

    # fc_stage = GatedStage("fc", 0, 0, 0, 1, 512, 2)
    full_stage = {"backbone_stages": backbone_stages, "initial": initial_stage}


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
    print(net)
    # summary(net, [(3, 368, 368), (1,)])
    x = torch.rand( 2, 3, 368, 368)
    y = net(Variable(x), torch.tensor(0.5))
    print(y.size())
    # print( y[0].size() )
    # y[0].backward
