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
import nnsearch.pytorch.torchx as torchx
from torchsummary import summary
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------

GatedVggStage = namedtuple("GatedVggStage",
                           ["nlayers", "nchannels", "ncomponents"])

GatedStage = namedtuple("GatedStage",
                           ["kernel_size", "stride", "padding", "nlayers", "nchannels", "ncomponents"])

def _Vgg_params(nlayers, nconv_stages, ncomponents, scale_ncomponents):
    channels = [64, 128, 256, 512, 512]
    assert (0 < nconv_stages <= len(channels))
    base = channels[0]
    fc_layers = 2
    fc_channels = 4096
    conv_params = []
    for i in range(len(channels)):
        scale = channels[i] // base if scale_ncomponents else 1
        conv_params.append(
            GatedVggStage(nlayers[i], channels[i], scale * ncomponents))
    fc_scale = fc_channels // base if scale_ncomponents else 1
    fc_params = GatedVggStage(fc_layers, fc_channels, fc_scale * ncomponents)
    return conv_params[:nconv_stages] + [fc_params]


def VggA(nconv_stages, ncomponents, scale_ncomponents=False):
    return _Vgg_params([1, 1, 2, 2, 2], nconv_stages, ncomponents, scale_ncomponents)


def VggB(nconv_stages, ncomponents, scale_ncomponents=False):
    return _Vgg_params([2, 2, 2, 2, 2], nconv_stages, ncomponents, scale_ncomponents)


def VggD(nconv_stages, ncomponents, scale_ncomponents=False):
    return _Vgg_params([2, 2, 3, 3, 3], nconv_stages, ncomponents, scale_ncomponents)


def VggE(nconv_stages, ncomponents, scale_ncomponents=False):
    return _Vgg_params([2, 2, 4, 4, 4], nconv_stages, ncomponents, scale_ncomponents)


# ----------------------------------------------------------------------------

class GatedMobilenet(GatedChainNetwork):
    """ A parameterizable VGG-style architecture.

    @article{simonyan2014very,
      title={Very deep convolutional networks for large-scale image recognition},
      author={Simonyan, Karen and Zisserman, Andrew},
      journal={arXiv preprint arXiv:1409.1556},
      year={2014}
    }
    """

    def __init__(self, gate, in_shape, nclasses, bb_stages, fc_stage,
                 initial_stage, refine_stage, batchnorm=False, dropout=0.5, **kwargs):
        # super().__init__()
        # self.gate = gate
        self.bb_stages = bb_stages
        self.fc_stage = fc_stage
        self.initial_stage = initial_stage
        self.refine_stage = refine_stage
        self.nclasses = nclasses
        self.batchnorm = batchnorm
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
                    if i > 0:
                        m, gated_m, in_shape = gatedDwConvBlock(stage.ncomponents, self.in_shape, self.in_channels, stage.nchannels, 
                            kernel_size=stage.kernel_size, padding=stage.padding)
                    else:
                        m, gated_m, in_shape = gatedConvBlock(stage.ncomponents, self.in_shape, self.in_channels, stage.nchannels,
                            kernel_size=stage.kernel_size, stride=stage.stride, padding=stage.padding, bias=True)
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


    def __set_fc(self):
        # FC layers
        self.in_shape = tuple([self.in_channels] + list(self.in_shape[1:]))
        self.in_channels = reduce(operator.mul, self.in_shape)
        for _ in range(self.fc_stage.nlayers):
            if self.fc_stage.ncomponents > 1:
                m = BlockGatedFullyConnected(self.fc_stage.ncomponents,
                                             self.in_channels, self.fc_stage.nchannels)
                self.modules.append(m)
                # self.gated_modules.append( (m, in_channels) )
                self.tmp_gated_modules.append((m, self.in_channels))
            else:
                self.modules.append(FullyConnected(
                    self.in_channels, self.fc_stage.nchannels))
            self.modules.append(nn.ReLU())
            if self.dropout > 0:
                self.modules.append(nn.Dropout(self.dropout))
            self.in_channels = self.fc_stage.nchannels

    def __set_classification_layer(self):
        self.modules.append(FullyConnected(self.in_channels, self.nclasses))

    def forward( self, x, u=None ):
        """
        Returns:
          `(y, gs)` where:
            `y`  : Network output
            `gs` : `[(g, info)]` List of things returned from `self.gate` in same
              order as `self.gated_modules`. `g` is the actual gate matrix, and
              `info` is any additional things returned (or `None`).
        """
        def expand( gout ):
          if isinstance(gout, tuple):
            g, info = gout # Fail fast on unexpected extra outputs
            return (g, info)
          else:
            return (gout, None)
        
        gs = []
        # FIXME: This is a hack for FasterRCNN integration. Find a better way.
        if u is None:
          self.gate.set_control( self._u )
        else:
          self.gate.set_control( u )
        # TODO: With the current architecture, set_control() has to happen before
        # reset() in case reset() needs to run the gate network. Should `u` be a
        # second parameter to reset()? Should it be an argument to gate() as well?
        # How do we support gate networks that require the outputs of arbitrary
        # layers in the data network in a modular way?
        self.gate.reset( x )
        for m in self.fn:
            # print(type(m))
            if isinstance(m, GatedModule):
                self.gate.next_module( m )
                g, info = expand( self.gate( x ) )
                gs.append( (g, info) )
                if self.normalize:
                    g = self._normalize( g )
                self._log_gbar( g )
                x = m( x, g )
            else:
                x = m( x )
            print(x.size())
            log.debug( "network.x: %s", x )
        return x, gs


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

    # order: "kernel_size", "stride", "padding", "nlayers", "nchannels", "ncomponents"
    backbone_stages = [GatedStage(3, 2, 0, 1,  32, 4), GatedStage(3, 1, 1, 1,  64, 4),
                       GatedStage(3, 2, 0, 1, 128, 4), GatedStage(3, 1, 1, 1, 128, 4),
                       GatedStage(3, 2, 0, 1, 256, 4), GatedStage(3, 1, 1, 1, 256, 4),
                       GatedStage(3, 1, 1, 1, 512, 4), GatedStage(3, 1, 2, 1, 512, 4),
                       GatedStage(3, 1, 1, 4, 512, 4)]

    initial_stage = [GatedStage(3, 1, 1, 3, 128, 4), GatedStage(1, 1, 0, 1, 512, 4),
                     GatedStage(1, 1, 0, 1, 21, 1)] 
    

    fc_stage = GatedVggStage(1, 512, 2)
    gate_modules = []

    for i, conv_stage in enumerate(backbone_stages):
        for _ in range(conv_stage.nlayers):
            # each stage uses depthwise conv2d with two gated layers except for the first conv stage
            if i > 0:
                count = strategy.PlusOneCount(strategy.UniformCount(conv_stage.ncomponents - 1))
                gate_modules.append(strategy.NestedCountGate(conv_stage.ncomponents, count))
            count = strategy.PlusOneCount(strategy.UniformCount(conv_stage.ncomponents - 1))
            gate_modules.append(strategy.NestedCountGate(conv_stage.ncomponents, count))

    for i, conv_stage in enumerate(initial_stage):
        for _ in range(conv_stage.nlayers):
            count = strategy.PlusOneCount(strategy.UniformCount(conv_stage.ncomponents - 1))
            gate_modules.append(strategy.NestedCountGate(conv_stage.ncomponents, count))

    for _ in range(fc_stage.nlayers):
        count = strategy.PlusOneCount(strategy.UniformCount(fc_stage.ncomponents - 1))
        gate_modules.append(strategy.NestedCountGate(fc_stage.ncomponents, count))

    gate = strategy.SequentialGate(gate_modules)

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

    net = GatedMobilenet(gate, (3, 32, 32), 21, backbone_stages, fc_stage, initial_stage, [])
    print(net)
    x = torch.rand( 2, 3, 32, 32)
    y = net(Variable(x), torch.tensor(0.5))
    print( y )
    y[0].backward
