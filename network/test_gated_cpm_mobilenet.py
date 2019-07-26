from collections import namedtuple
from functools import reduce
import logging
import operator

import torch
import torch.nn as nn
from nnsearch.pytorch.data import datasets
import nnsearch.pytorch.gated.vgg as vgg
from nnsearch.pytorch.gated.module import (BlockGatedConv2d,
                                           BlockGatedFullyConnected, GatedChainNetwork)
import nnsearch.pytorch.gated.strategy as strategy
from network.gated_cpm_mobilenet import Gated_CPM_MobileNet
from nnsearch.pytorch.modules import FullyConnected
import nnsearch.pytorch.torchx as torchx

log = logging.getLogger(__name__)
from torchsummary import summary
# ----------------------------------------------------------------------------

GatedVggStage = namedtuple("GatedVggStage",
                           ["nlayers", "nchannels", "ncomponents"])


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

    conv_stages = [
        GatedVggStage(2, 64, 8), GatedVggStage(3, 128, 8), GatedVggStage(3, 256, 16)]
    fc_stage = GatedVggStage(2, 4096, 16)
    gate_modules = []


    # for conv_stage in conv_stages:
    # for _ in range(conv_stage.nlayers):
    # count = strategy.PlusOneCount( strategy.UniformCount( conv_stage.ncomponents - 1 ) )
    # gate_modules.append( strategy.NestedCountGate( conv_stage.ncomponents, count ) )
    # for _ in range(fc_stage.nlayers):
    # count = strategy.PlusOneCount( strategy.UniformCount( fc_stage.ncomponents - 1 ) )
    # gate_modules.append( strategy.NestedCountGate( fc_stage.ncomponents, count ) )
    # gate = strategy.SequentialGate( gate_modules )
    def make_static_gate_module(n, min_active=0):
        # log.debug( "make_static_gate_module.n: %s", n )
        glayer = []
        # if args.granularity == "count":
        glayer.append(strategy.ProportionToCount(min_active, n))
        glayer.append(strategy.CountToNestedGate(n))
        # if args.order == "permutation":
        # glayer.append( strategy.PermuteColumns() )
        # elif args.order != "nested":
        # parser.error( "--order={}".format(args.order) )
        # else:
        # parser.error( "--granularity={}".format(args.granularity) )
        return strategy.StaticGate(nn.Sequential(*glayer))


    def make_gate(dataset, ncomponents, in_channels=None, always_on=False):
        # assert( all( c == ncomponents[0] for c in ncomponents[1:] ) )
        if in_channels is not None:
            assert len(ncomponents) == len(in_channels)
        # print( in_channels )

        # if args.granularity == "count":
        noptions = [n + 1 for n in ncomponents]
        # else:
        # noptions = ncomponents[:]

        # control_tokens = args.control.split( "," )
        # control = control_tokens[0]
        # if control == "static":
        # gate_modules = [make_gate_controller(c, always_on=always_on)
        # for c in ncomponents]
        # gate = strategy.SequentialGate( gate_modules )
        min_active = 1 if always_on else 0
        ms = [make_static_gate_module(n, min_active) for n in ncomponents]
        return strategy.SequentialGate(ms)


    def gated_vgg(dataset, arch_string):
        from nnsearch.pytorch.gated.vgg import VggA, VggB, VggD, VggE
        stages = eval(arch_string)
        stages = [GatedVggStage(*t) for t in stages]

        ncomponents = [stage.ncomponents for stage in stages]
        in_channels = [dataset.in_shape[0]] + [stage.nchannels for stage in stages[:-1]]
        gate = make_gate(dataset, ncomponents, in_channels, always_on=False)
        chunk_sizes = [stage.nlayers for stage in stages]
        gate = strategy.GateChunker(gate, chunk_sizes)

        conv_stages, fc_stage = stages[:-1], stages[-1]
        net = vgg.GatedVgg(gate, dataset.in_shape, dataset.nclasses, conv_stages,
                       fc_stage, batchnorm=True, dropout=float(0.8),
                       reverse_normalize=True)
        return net


    # groups = []
    # gi = 0
    # for conv_stage in conv_stages:
    #   count = strategy.PlusOneCount( strategy.UniformCount( conv_stage.ncomponents - 1 ) )
    #   gate_modules.append( strategy.NestedCountGate( conv_stage.ncomponents, count ) )
    #   groups.extend( [gi] * conv_stage.nlayers )
    #   gi += 1
    # count = strategy.PlusOneCount( strategy.UniformCount( fc_stage.ncomponents - 1 ) )
    # gate_modules.append( strategy.NestedCountGate( fc_stage.ncomponents, count ) )
    # groups.extend( [gi] * fc_stage.nlayers )
    # gate = strategy.GroupedGate( gate_modules, groups )

    # net = GatedVgg( gate, (3, 32, 32), 10, conv_stages, fc_stage )
    print(datasets)
    net = gated_vgg(datasets['cifar10'], "(2,64,8),(3,128,8), (3,256, 16)")
    print(net)
    x = torch.rand(4, 3, 32, 32)
    y = net(x, torch.tensor([0.5]))
    # print(y)
    # y[0].backward