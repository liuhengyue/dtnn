from collections import namedtuple
from functools import reduce
import logging
import operator
import math
import torch
import torch.nn as nn
from torchsummary import summary
from nnsearch.pytorch.gated.module import (BlockGatedConv3d, BlockGatedConv2d, BlockGatedFullyConnected,
    GatedChainNetwork, GatedModule)
from modules.conv import gatedConvBlock, gatedDwConvBlock, gated3dConvBlock, Maxpool3dWrapper
import nnsearch.pytorch.gated.strategy as strategy
from nnsearch.pytorch.modules import FullyConnected
import shape_flop_util as util
import nnsearch.pytorch.torchx as torchx
#from torchsummary import summary
log = logging.getLogger(__name__)
from functools import reduce, singledispatch
from modules.utils import make_sequentialGate
# ----------------------------------------------------------------------------

GatedStage = namedtuple("GatedStage",
                           ["name", "kernel_size", "stride", "padding", "nlayers", "nchannels", "ncomponents"])


# ----------------------------------------------------------------------------
@singledispatch
def output_shape(layer, input_shape):
    """ Computes the output shape given a layer and input shape, without
    evaluating the layer. Raises `NotImplementedError` for unsupported layer
    types.

    Parameters:
        `layer` : The layer whose output shape is desired
        `input_shape` : The shape of the input, in the format (N, H, W, ...). Note
            that this must *not* include a "batch" dimension or anything similar.
    """
    raise NotImplementedError(layer)

# @output_shape.register(nn.Softmax)
# @output_shape.register(nn.BatchNorm2d)
@output_shape.register(nn.ReLU)
@output_shape.register(nn.Dropout)
@output_shape.register(nn.BatchNorm3d)
def _(layer, input_shape):
   return input_shape


@output_shape.register(nn.Linear)
def _(layer, input_shape):
    assert (flat_size(input_shape) == layer.in_features)
    return tuple([layer.out_features])

@output_shape.register(nn.MaxPool3d)
def _(layer, input_shape):
    out_channels = input_shape[0]
    return _output_shape_Conv(3, input_shape, out_channels,
                              layer.kernel_size, layer.stride, layer.padding, layer.dilation, False)

@output_shape.register(nn.Conv3d)
def _(layer, input_shape):
    return _output_shape_Conv(3, input_shape, layer.out_channels,
                              layer.kernel_size, layer.stride, layer.padding, layer.dilation, False)

@output_shape.register(BlockGatedConv3d)
def _(layer, input_shape):
    out_channels = input_shape[0]
    # print("ssss", dir(layer.components[0]).out_channel)
    out_channels= layer.components[0].out_channels * len(layer.components)
    return _output_shape_Conv(3, input_shape, out_channels,
                              layer.components[0].kernel_size, layer.components[0].stride, layer.components[0].padding, layer.components[0].dilation, False)

@output_shape.register(BlockGatedFullyConnected)
def _(layer, input_shape):
    out_channels = input_shape[0]
    # print("ssss", dir(layer.components[0]).out_channel)
    out_channels = layer.components[0].out_features * len(layer.components)
    return tuple([out_channels])


def _output_shape_Conv(dim, input_shape, out_channels, kernel_size, stride,
                       padding, dilation, ceil_mode):
    """ Implements output_shape for "conv-like" layers, including pooling layers.
    """
    assert (len(input_shape) == dim + 1)
    kernel_size = _maybe_expand_tuple(dim, kernel_size)
    stride = _maybe_expand_tuple(dim, stride)
    padding = _maybe_expand_tuple(dim, padding)
    dilation = _maybe_expand_tuple(dim, dilation)
    quantize = math.ceil if ceil_mode else math.floor
    out_dim = [quantize(
        (input_shape[i + 1] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)
        / stride[i] + 1)
        for i in range(dim)]
    output_shape = tuple([out_channels] + out_dim)
    return output_shape

def flat_size(shape):
    return reduce(operator.mul, shape)

def _maybe_expand_tuple(dim, tuple_or_int):
    if type(tuple_or_int) is int:
        tuple_or_int = tuple([tuple_or_int] * dim)
    else:
        assert (type(tuple_or_int) is tuple)
    return tuple_or_int
# --------------------------------------------------------------------------------------


class GatedC3D(GatedChainNetwork):
    """ A parameterizable conv 3d architecture.
    """

    def __init__(self, gate, in_shape, nclasses, c3d_stages, fc_stages,
                 batchnorm=True, dropout=0.5, **kwargs):
        # in_shape: (C, D, H, W), D is time/sequence

        self.fc_stages = fc_stages
        self.c3d_stages = c3d_stages
        self.nclasses = nclasses
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.tmp_modules = []
        self.tmp_gated_modules = [] # GatedChainNetwork has a property called gated_modules
        
        self.__set_c3d_model()
        self.__set_fc()
        self.__set_classification_layer()
        # self.fn = nn.ModuleList( modules )
        # print("modules------------------",modules)
        # print("gated modules------------------", gated_modules)
        super().__init__(gate, self.tmp_modules, self.tmp_gated_modules, **kwargs)


    def __set_c3d_model(self):
        for i, stage in enumerate(self.c3d_stages):
            for _ in range(stage.nlayers):
                # conv or pool
                if stage.name == "conv":
                    if stage.ncomponents > 1:
                        m, gated_m, in_shape = gated3dConvBlock(stage.ncomponents, self.in_shape, self.in_channels, stage.nchannels,
                                                kernel_size=stage.kernel_size, stride=stage.stride, padding=stage.padding, bias=True)
                        self.in_shape = in_shape
                        self.tmp_modules.extend(m)
                        self.tmp_gated_modules.extend(gated_m)
                    else:
                        self.tmp_modules.append(nn.Conv3d(self.in_channels, stage.nchannels,
                                                          kernel_size=stage.kernel_size, stride=stage.stride, padding=stage.padding))
                        # for first layer, add bn and relu
                        self.tmp_modules.append(nn.BatchNorm3d(stage.nchannels))
                        self.tmp_modules.append(nn.ReLU())
                    self.in_channels = stage.nchannels
                elif stage.name == "pool":
                    pool, in_shape = Maxpool3dWrapper(self.in_shape, kernel_size=stage.kernel_size, stride=stage.stride)
                    self.tmp_modules.extend(pool)
                    self.in_shape = in_shape
                    # compute 3dpool shape
                # print(self.in_shape)
                
                


    def __set_fc(self): 
        # FC layers
        self.in_shape = tuple([self.in_channels] + list(self.in_shape[1:]))
        self.in_channels = reduce(operator.mul, self.in_shape)
        for fc_stage in self.fc_stages:
            for _ in range(fc_stage.nlayers):
                if fc_stage.ncomponents > 1:
                    m = BlockGatedFullyConnected(fc_stage.ncomponents,
                                                 self.in_channels, fc_stage.nchannels)
                    self.tmp_modules.append(m)
                    # self.gated_modules.append( (m, in_channels) )
                    self.tmp_gated_modules.append((m, self.in_channels))
                else:
                    self.tmp_modules.append(FullyConnected(
                        self.in_channels, fc_stage.nchannels))
                self.tmp_modules.append(nn.ReLU())
                if self.dropout > 0:
                    self.tmp_modules.append(nn.Dropout(self.dropout))
                self.in_channels = fc_stage.nchannels

    def __set_classification_layer(self):
        self.tmp_modules.append(FullyConnected(self.in_channels, self.nclasses))

    def flops(self, in_shape):
        total_macc = 0
        gated_macc = []
        # Keep in mind that this is only calculating flops for the function part of
        # the network as the "work" for the gate is not a network atm
        for (i, m) in enumerate(self.fn):
            if isinstance(m, BlockGatedConv3d):
                # print(dir(m))
                module_macc = []
                for c in m.components:
                    module_macc.append(util.flops(c, in_shape))
                    #in_shape = output_shape(c, in_shape)
                gated_macc.append(module_macc)
                in_shape = output_shape(m, in_shape)
                #print("CALCULATING SHAPE", in_shape, m)
                #print(module_macc)
            elif isinstance(m, BlockGatedFullyConnected):
                module_macc = []
                for c in m.components:
                    module_macc.append(util.flops(c, in_shape))
                gated_macc.append(module_macc)
                in_shape = util.output_shape(m, in_shape)
                #print("CALCULATING SHAPE", in_shape)
            else:
                total_macc += util.flops(m, in_shape).macc
                in_shape = output_shape(m, in_shape)
                #print("CALCULATING SHAPE", in_shape)

        total_macc += sum(sum(c.macc for c in m) for m in gated_macc)
        print("TOTAL FLOPS", total_macc)
        return (total_macc, gated_macc)

def C3dDataNetwork(in_shape=(3, 16, 100, 160), num_classes=27, gate_during_eval=True):
    span_factor = 1

    c3d_stages = [GatedStage("conv", 3, 1, 1, 1, 64 * span_factor, 1), GatedStage("pool", (1, 2, 2), (1, 2, 2), 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 1, 128 * span_factor, 16), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 2, 256 * span_factor, 32), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 2, 512 * span_factor, 32), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                  GatedStage("conv", 3, 1, 1, 2, 512 * span_factor, 32), GatedStage("pool", 2, 2, 0, 1, 0, 0), ]

    fc_stages = [GatedStage("fc", 0, 0, 0, 2, 512 * span_factor, 16)]

    # non gated
    # c3d_stages = [GatedStage("conv", 3, 1, 1, 1, 64, 1), GatedStage("pool", (1, 2, 2), (1, 2, 2), 0, 1, 0, 0),
    #               GatedStage("conv", 3, 1, 1, 1, 128, 1), GatedStage("pool", 2, 2, 0, 1, 0, 0),
    #               GatedStage("conv", 3, 1, 1, 2, 256, 1), GatedStage("pool", 2, 2, 0, 1, 0, 0),
    #               GatedStage("conv", 3, 1, 1, 2, 512, 1), GatedStage("pool", 2, 2, 0, 1, 0, 0),
    #               GatedStage("conv", 3, 1, 1, 2, 512, 1), GatedStage("pool", 2, 2, 0, 1, 0, 0), ]
    #
    # fc_stages = [GatedStage("fc", 0, 0, 0, 2, 512, 1)]

    stages = {"c3d": c3d_stages, "fc": fc_stages}
    gate = make_sequentialGate(stages, gate_during_eval=gate_during_eval)
    # in_shape = (21, 16, 45, 45)
    # in_shape = (3, 16, 368, 368) # for raw input
    c3d_pars = {"c3d": c3d_stages, "fc": fc_stages, "gate": gate,
            "in_shape": in_shape, "num_classes": num_classes}

    c3d_net = GatedC3D(c3d_pars["gate"], c3d_pars["in_shape"],
                       c3d_pars["num_classes"], c3d_pars["c3d"], c3d_pars["fc"], dropout=0)

    return c3d_net
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


    net = C3dDataNetwork(in_shape=(3,16,100,160), gate_during_eval=False).cuda()
    net.eval()
    net.flops((3, 16, 100, 160))
    # print(net)

    summary(net, [(3, 16, 100, 160), (1,)], device="cuda")
    x = torch.rand(1, 3, 16, 100, 160).cuda()
    u = torch.tensor(1.0).cuda()
    y, gs = net(x, u)
    print("output size: {} \n gate size: {} ".format(y.size(), len(gs)))
    print("gate matrix: \n {}".format([g[0] for g in gs]))
    y.backward
    print("BP works.")
    # u = 0.5
    # for gated_module in net._gated_modules:
    #     n = len(gated_module[0].components)
    #     c = int(n * u)
    #     for i in range(c):
    #         gated_module[0].components[i].requires_grad = False
