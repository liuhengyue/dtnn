from   collections import namedtuple
from   functools import reduce
import logging
import operator

import torch
import torch.nn as nn

from   nnsearch.pytorch.gated.module import (BlockGatedConv2d, 
  BlockGatedFullyConnected, GatedChainNetwork)
import nnsearch.pytorch.gated.strategy as strategy
from   nnsearch.pytorch.modules import FullyConnected
import nnsearch.pytorch.torchx as torchx

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

GatedVggStage = namedtuple( "GatedVggStage",
  ["nlayers", "nchannels", "ncomponents"] )
  
def _Vgg_params( nlayers, nconv_stages, ncomponents, scale_ncomponents ):
  channels = [64, 128, 256, 512, 512]
  assert( 0 < nconv_stages <= len(channels) )
  base = channels[0]
  fc_layers = 2
  fc_channels = 4096
  conv_params = []
  for i in range(len(channels)):
    scale = channels[i] // base if scale_ncomponents else 1
    conv_params.append(
      GatedVggStage(nlayers[i], channels[i], scale*ncomponents) )
  fc_scale = fc_channels // base if scale_ncomponents else 1
  fc_params = GatedVggStage(fc_layers, fc_channels, fc_scale*ncomponents)
  return conv_params[:nconv_stages] + [fc_params]
  
def VggA( nconv_stages, ncomponents, scale_ncomponents=False ):
  return _Vgg_params( [1, 1, 2, 2, 2], nconv_stages, ncomponents, scale_ncomponents )
  
def VggB( nconv_stages, ncomponents, scale_ncomponents=False ):
  return _Vgg_params( [2, 2, 2, 2, 2], nconv_stages, ncomponents, scale_ncomponents )

def VggD( nconv_stages, ncomponents, scale_ncomponents=False ):
  return _Vgg_params( [2, 2, 3, 3, 3], nconv_stages, ncomponents, scale_ncomponents )
  
def VggE( nconv_stages, ncomponents, scale_ncomponents=False ):
  return _Vgg_params( [2, 2, 4, 4, 4], nconv_stages, ncomponents, scale_ncomponents )

# ----------------------------------------------------------------------------

class GatedVgg(GatedChainNetwork):
  """ A parameterizable VGG-style architecture.
  
  @article{simonyan2014very,
    title={Very deep convolutional networks for large-scale image recognition},
    author={Simonyan, Karen and Zisserman, Andrew},
    journal={arXiv preprint arXiv:1409.1556},
    year={2014}
  }
  """
  def __init__( self, gate, in_shape, nclasses, conv_stages, fc_stage,
                batchnorm=False, dropout=0.5, **kwargs ):
    # super().__init__()
    # self.gate = gate
    modules = []
    kernel_size = 3
    pool_size = 2
    in_channels = in_shape[0]
    # self.gated_modules = []
    gated_modules = []
    # Convolution layers
    for stage in conv_stages:
      for _ in range(stage.nlayers):
        if stage.ncomponents > 1:
          m = BlockGatedConv2d( stage.ncomponents,
            in_channels, stage.nchannels, kernel_size, padding=1 )
          modules.append( m )
          shape = tuple([in_channels]) + in_shape[1:]
          # self.gated_modules.append( (m, shape) )
          gated_modules.append( (m, shape) )
        else:
          modules.append( nn.Conv2d( 
            in_channels, stage.nchannels, kernel_size, padding=1 ) )
        if batchnorm:
          modules.append( nn.BatchNorm2d( stage.nchannels ) )
        modules.append( nn.ReLU() )
        in_channels = stage.nchannels
      pool = nn.MaxPool2d( pool_size )
      modules.append( pool )
      in_shape = torchx.output_shape( pool, in_shape )
    # FC layers
    in_shape = tuple( [in_channels] + list(in_shape[1:]) )
    in_channels = reduce( operator.mul, in_shape )
    for _ in range(fc_stage.nlayers):
      if fc_stage.ncomponents > 1:
        m = BlockGatedFullyConnected( fc_stage.ncomponents,
          in_channels, fc_stage.nchannels )
        modules.append( m )
        # self.gated_modules.append( (m, in_channels) )
        gated_modules.append( (m, in_channels) )
      else:
        modules.append( FullyConnected(
          in_channels, fc_stage.nchannels ) )
      modules.append( nn.ReLU() )
      if dropout > 0:
        modules.append( nn.Dropout( dropout ) )
      in_channels = fc_stage.nchannels
    # Classification layer
    modules.append( FullyConnected( in_channels, nclasses ) )
    # self.fn = nn.ModuleList( modules )
    super().__init__( gate, modules, gated_modules, **kwargs )
    
# ----------------------------------------------------------------------------

if __name__ == "__main__":
  import torch
  from torch.autograd import Variable
  
  import nnsearch.logging as mylog
  
  # Logger setup
  mylog.add_log_level( "VERBOSE", logging.INFO - 5 )
  root_logger = logging.getLogger()
  root_logger.setLevel( logging.DEBUG )
  # Need to set encoding or Windows will choke on ellipsis character in
  # PyTorch tensor formatting
  handler = logging.FileHandler( "vgg.log", "w", "utf-8")
  handler.setFormatter( logging.Formatter("%(levelname)s:%(name)s: %(message)s") )
  root_logger.addHandler(handler)
  
  conv_stages = [
    GatedVggStage(2, 64, 8), GatedVggStage(3, 128, 8), GatedVggStage(3, 256, 16) ]
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
  
  groups = []
  gi = 0
  for conv_stage in conv_stages:
    count = strategy.PlusOneCount( strategy.UniformCount( conv_stage.ncomponents - 1 ) )
    gate_modules.append( strategy.NestedCountGate( conv_stage.ncomponents, count ) )
    groups.extend( [gi] * conv_stage.nlayers )
    gi += 1
  count = strategy.PlusOneCount( strategy.UniformCount( fc_stage.ncomponents - 1 ) )
  gate_modules.append( strategy.NestedCountGate( fc_stage.ncomponents, count ) )
  groups.extend( [gi] * fc_stage.nlayers )
  gate = strategy.GroupedGate( gate_modules, groups )
  
  net = GatedVgg( gate, (3, 32, 32), 10, conv_stages, fc_stage )
  print( net )
  x = torch.rand( 4, 3, 32, 32 )
  y = net(Variable(x))
  print( y )
  y.backward
