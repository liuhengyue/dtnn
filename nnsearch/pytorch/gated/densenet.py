from   collections import namedtuple
import logging
import math

import torch
import torch.nn as nn

import nnsearch.pytorch.data as data
from   nnsearch.pytorch.gated.module import GatedChainNetwork, GatedModule, GatedModuleList
from   nnsearch.pytorch.modules import FullyConnected, GlobalAvgPool2d
import nnsearch.pytorch.torchx as torchx
import nnsearch.pytorch.gated.strategy as strategy
log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

DenseNetBlockSpec = namedtuple( "DenseNetBlockSpec", ["nlayers", "bottleneck"] )

class GatedDenseNetBlock(GatedModule, nn.Module):
  def __init__( self, k, n, in_channels, bottleneck ):
    super().__init__()
    components = []
    self.in_channels = in_channels
    self.growth_rate = k
    for i in range(n):
      layer = []
      if bottleneck:
        layer.append( nn.BatchNorm2d( in_channels ) )
        layer.append( nn.ReLU() )
        layer.append( nn.Conv2d( in_channels, 4*k, 1, bias=False ) )
        in_channels = 4*k
      layer.append( nn.BatchNorm2d( in_channels ) )
      layer.append( nn.ReLU() )
      layer.append( nn.Conv2d( in_channels, k, 3, padding=1, bias=False ) )
      components.append( nn.Sequential( *layer ) )
      in_channels = self.in_channels + (i+1) * k
    self.out_channels = in_channels
    self.components = nn.ModuleList( components )
    
  def forward( self, x, g ):
    for (i, m) in enumerate(self.components):
      y = m(x)
      gi = g[:,i].contiguous()
      gu = torchx.unsqueeze_right_as( gi, y )
      y = gu * y
      x = torch.cat([x, y], dim=1)
    return x
  
  @property
  def ncomponents( self ):
    return len(self.components)
  
  @property
  def vectorize( self ):
    return self._vectorize
    
  @vectorize.setter
  def vectorize( self, value ):
    self._vectorize = value
    
@torchx.output_shape.register(GatedDenseNetBlock)
def _( layer, in_shape ):
  out_channels = layer.in_channels + len(layer.components) * layer.growth_rate
  return (out_channels,) + in_shape[1:]
    
# ----------------------------------------------------------------------------

DenseNetDefaults = namedtuple("DenseNetDefaults", ["input", "in_shape"])

def defaults( dataset, growth_rate, compression ):
  if isinstance(dataset, (data.Cifar10Dataset, data.SvhnDataset)):
    in_channels = 2*growth_rate if compression < 1.0 else 16
    # We do not do BN/ReLU on the input because DenseNet uses pre-activation,
    # so it will happen in the first dense block anyway
    input = nn.Conv2d( dataset.in_shape[0], in_channels, 3, padding=1, bias=False )
    in_shape = torchx.output_shape( input, dataset.in_shape )
  elif isinstance(dataset, data.ImageNetDataset):
    in_channels = 2 * growth_rate
    input = nn.Sequential(
      nn.Conv2d( dataset.in_shape[0], in_channels, 7,
                 stride=2, padding=3, bias=False ),
      nn.BatchNorm2d( in_channels ),
      nn.ReLU(),
      nn.MaxPool2d( 3, stride=2, padding=1 ) )
    in_shape = torchx.output_shape( input, dataset.in_shape )
  else:
    raise NotImplementedError()
  return DenseNetDefaults(input, in_shape)

# FIXME: We needed a way to calculate input sizes for learned gating layers, so
# we duplicated some of the code from GatedDenseNet.__init__. We should be
# calling in_channels() from GatedDenseNet.__init__ to avoid duplication.
def in_channels( k, in_shape, dense_blocks, compression=1.0 ):
  in_ch = in_shape[0]
  result = []
  for (i, spec) in enumerate(dense_blocks):
    if i > 0: # Dimension reduction
      in_ch = math.floor( in_ch * compression )
    result.append( in_ch )
    in_ch += k * spec.nlayers
  return result
  
# ----------------------------------------------------------------------------

class GatedDenseNet(GatedChainNetwork):
  def __init__( self, gate, k, input, in_shape, nclasses, 
                dense_blocks, compression=1.0, **kwargs ):
    """
    Parameters:
      `gate` : Gating module
      `k` : "growth rate" parameter
      `input` : An arbitrary input module
      `in_shape` : The shape of the *output* of the `input` module
      `nclasses` : Number of output classes
      `dense_blocks` : `[DenseNetBlockSpec]`
      `compression` : Compression rate in [0,1]
    """
    assert( compression > 0 )
    assert( compression <= 1 )
    
    gated_modules = []
    # Input
    modules = [input]
    in_channels = in_shape[0]
    # Dense blocks
    for (i, spec) in enumerate(dense_blocks):
      if i > 0: # Dimension reduction
        out_channels = math.floor( in_channels * compression )
        modules.append( nn.BatchNorm2d( in_channels ) )
        modules.append( nn.ReLU() )
        modules.append( nn.Conv2d( in_channels, out_channels, 1, bias=False ) )
        pool = nn.AvgPool2d( 2 )
        modules.append( pool )
        in_shape = torchx.output_shape( pool, in_shape )
        in_channels = out_channels
      m = GatedDenseNetBlock(
        k, spec.nlayers, in_channels, spec.bottleneck )
      shape = tuple([in_channels]) + in_shape[1:]
      modules.append( m )
      gated_modules.append( (m, shape) )
      in_channels += k * spec.nlayers
    # Classification
    modules.append( nn.BatchNorm2d( in_channels ) )
    modules.append( nn.ReLU() )
    modules.append( GlobalAvgPool2d() )
    modules.append( FullyConnected( in_channels, nclasses ) )
    super().__init__( gate, modules, gated_modules, **kwargs )
    
  def flops( self, in_shape ):
    total_macc = 0
    gated_macc = []
    for (i, m) in enumerate(self.fn):
      if isinstance(m, GatedDenseNetBlock):
        module_macc = []
        for c in m.components:
          module_macc.append( torchx.flops( c, in_shape ) )
          # in_shape = torchx.output_shape( c, in_shape )
        gated_macc.append( module_macc )
        in_shape = torchx.output_shape( m, in_shape )
      else:
        total_macc += torchx.flops( m, in_shape ).macc
        in_shape = torchx.output_shape( m, in_shape )
    total_macc += sum( sum(c.macc for c in m) for m in gated_macc )
    return (total_macc, gated_macc)

# ----------------------------------------------------------------------------
  
class GatedDenseNet_Old(nn.Module):
  def __init__( self, gate, k, input, in_shape, nclasses, 
                dense_blocks, compression=1.0 ):
    """
    Parameters:
      `gate` : Gating module
      `k` : "growth rate" parameter
      `input` : An arbitrary input module
      `in_shape` : The shape of the *output* of the `input` module
      `nclasses` : Number of output classes
      `dense_blocks` : `[DenseNetBlockSpec]`
      `compression` : Compression rate in [0,1]
    """
    assert( compression > 0 )
    assert( compression <= 1 )
    super().__init__()
    self.gate = gate
    self.gated_modules = []
    # Input
    modules = [input]
    in_channels = in_shape[0]
    # Dense blocks
    for (i, spec) in enumerate(dense_blocks):
      if i > 0: # Dimension reduction
        out_channels = math.floor( in_channels * compression )
        modules.append( nn.BatchNorm2d( in_channels ) )
        modules.append( nn.ReLU() )
        modules.append( nn.Conv2d( in_channels, out_channels, 1, bias=False ) )
        pool = nn.AvgPool2d( 2 )
        modules.append( pool )
        in_shape = torchx.output_shape( pool, in_shape )
        in_channels = out_channels
      m = GatedDenseNetBlock(
        k, spec.nlayers, in_channels, spec.bottleneck )
      shape = tuple([in_channels]) + in_shape[1:]
      modules.append( m )
      self.gated_modules.append( (m, shape) )
      in_channels += k * spec.nlayers
    # Classification
    modules.append( nn.BatchNorm2d( in_channels ) )
    modules.append( nn.ReLU() )
    modules.append( nn.AvgPool2d( tuple(in_shape[1:]) ) ) # Global pool
    modules.append( FullyConnected( in_channels, nclasses ) )
    self.fn = nn.ModuleList( modules )
    
  def forward( self, x ):
    gs = []
    self.gate.reset()
    for m in self.fn:
      if isinstance(m, GatedModule):
        self.gate.next_module( m )
        g = self.gate( x )
        gs.append( g )
        # Normalize by count
        c = torch.sum( g, dim=1, keepdim=True )
        z = c + 1e-12 # Avoid divide-by-zero
        gbar = g / z
        log.debug( "densenet.gbar: %s", gbar )
        x = m( x, gbar )
      else:
        x = m( x )
    log.debug( "densenet.x: %s", x )
    return x, gs
    
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
  handler = logging.FileHandler( "densenet.log", "w", "utf-8")
  handler.setFormatter( logging.Formatter("%(levelname)s:%(name)s: %(message)s") )
  root_logger.addHandler(handler)
  
  k = 3
  input = nn.Conv2d( 3, 16, 3, padding=1, bias=False )
  in_shape = (16, 32, 32)
  dense_blocks = [
    DenseNetBlockSpec(4, False), DenseNetBlockSpec(6, False),
    DenseNetBlockSpec(8, True) ]
  gate_modules = []
  for spec in dense_blocks:
    log.debug( "block: %s; nlayers = %s", spec, spec.nlayers )
    gm = strategy.NestedCountGate( spec.nlayers, strategy.UniformCount( spec.nlayers ) )
    log.debug( "gm: %s", gm )
    gate_modules.append( gm )
  gate = strategy.SequentialGate( gate_modules )
  net = GatedDenseNet( gate, k, input, in_shape, 10, dense_blocks, compression=0.5 )
  # print( net )
  x = torch.rand( 4, 3, 32, 32 )
  y = net(Variable(x), torch.tensor(0.5))
  print( y )
  y[0].backward
