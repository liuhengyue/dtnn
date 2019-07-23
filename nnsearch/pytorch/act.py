import abc
import logging

import torch
from   torch.autograd import Variable
import torch.nn as nn

from   nnsearch.pytorch.models.resnext import ResNeXtBlock, ResNeXtBottleneckPath
from   nnsearch.pytorch.modules import FullyConnected
import nnsearch.pytorch.torchx as torchx

log = logging.getLogger( __name__ )

class ACTNetwork(metaclass=abc.ABCMeta):
  @property
  @abc.abstractmethod
  def halt_modules( self ):
    return NotImplemented

class ACTChainNetwork(nn.ModuleList, ACTNetwork):
  def __init__( self, modules ):
    super().__init__( modules )
  
  @property
  def halt_modules( self ):
    for m in self.modules():
      if isinstance(m, ACTBlock):
        for h in m.halt_modules:
          yield h
  
  def forward( self, x ):
    batch_size = x.size(0)
    rho = Variable(torch.zeros(batch_size, 1).type_as(x.data))
    act_block = 0
    for m in self:
      if isinstance(m, ACTBlock):
        x, rho_i = m( x )
        rho += rho_i
        log.debug( "act.block.%s: rho_i = %s; rho = %s", act_block, rho_i, rho )
        act_block += 1
      else:
        x = m( x )
    return x, rho

class ACTBlock(nn.Module, metaclass=abc.ABCMeta):
  """ A generalized implementation of the Adaptive Computation Time (ACT)
  method.
  """
  def __init__( self, data_modules, halt_modules, eps=1e-2 ):
    super().__init__()
    
    self.data_modules = data_modules
    self.halt_modules = halt_modules
    self.eps = eps
    
    for (i, m) in enumerate(data_modules):
      self.add_module( "F{}".format(i), m )
    for (i, m) in enumerate(halt_modules):
      self.add_module( "H{}".format(i), m )
  
  @abc.abstractmethod
  def next_input( self, stage_idx, input, module_output ):
    """ Computes the input to the next module, given the input to the ACT
    block and the output of the previous module.
    """
    return NotImplemented
  
  @abc.abstractmethod
  def epilog( self, input, output ):
    """ Computes the final ACT block output given the ACT block input and the
    current ACT block output.
    """
    return NotImplemented
  
  def forward( self, x ):
    # We implement a vectorized batch version of the ACT forward step. The
    # original code is formulated for one example at a time.
    batch_size = x.size(0)
    c   = 0 * Variable(torch.ones(batch_size, 1).type_as(x.data))
    R   = 1 * Variable(torch.ones(batch_size, 1).type_as(x.data))
    rho = 0 * Variable(torch.ones(batch_size, 1).type_as(x.data))
    mask = (c < 1.0 - self.eps).float()
    module_input = x
    output = None
    for (i, m) in enumerate(self.data_modules):
      log.debug( "act.module %s", i )
      module_output = m.forward( module_input )
      module_input = self.next_input( i, x, module_output )
      if i < len(self.data_modules) - 1:
        h = self.halt_modules[i]( module_output )
      else:
        h = Variable(torch.ones(batch_size, 1).type_as(x.data))
      log.debug( "act.h.%s: %s", i, h )
      c += h
      rho += mask
      
      mask = (c < 1.0 - self.eps).float()
      log.debug( "act.mask.%s: %s", i, mask )
      masku = torchx.unsqueeze_right_as( mask, module_output )
      finished = (torch.sum(mask) == 0).all()
      
      if output is None:
        output = torch.zeros_like(module_output)
      # Non-halted modules
      hu = torchx.unsqueeze_right_as( h, module_output )
      output += masku * hu * module_output
      R = R - mask * h 
      log.debug( "act.R.%s: %s", i, R )
      # Halted modules
      Ru = torchx.unsqueeze_right_as( R, module_output )
      output += (1 - masku) * Ru * module_output
      rho    += (1 - mask) * R
      log.debug( "act.rho.%s: %s", i, rho )
      
      if finished:
        break
      
    output = self.epilog( x, output )
    return output, rho
    
class SequentialACT(ACTBlock):
  """ Implements ACT where the modules form a chain. (This is the version from
  the original paper).
  """
  def __init__( self, data_modules, halt_modules, eps=1e-2 ):
    super().__init__( data_modules, halt_modules, eps=eps )
    
  def next_input( self, stage_idx, input, module_output ):
    # Next module receives direct input from previous module
    return module_output
  
  def epilog( self, input, output ):
    # Nothing special to do at the end
    return output
    
class ParallelResNeXtACT(ACTBlock):
  """ Implements ACT where the modules are parallel ResNeXt paths.
  """
  def __init__( self, halt_modules, in_features, internal_features, npaths, stride=1, expansion=2, eps=1e-2 ):
    """
    ResNeXt "bottleneck block", divided into separately-gated "paths" (Figure 3a
    in the paper).
    
    Parameters:
      `in_features`: Number of input features to the block.
      `internal_features`: Number of output features of internal layers.
      `npaths`: Number of separate paths in the block. Must be a factor of `internal_features`.
      `stride`: If > 1, the first layer in the block performs convolutional down-sampling.
      `expansion`: Number of output features is `internal_features*expansion`.
    """
    assert( internal_features % npaths == 0 )
    path_features = internal_features // npaths
    self.out_features = internal_features*expansion
        
    data_modules = [ResNeXtBottleneckPath(in_features, path_features, self.out_features, stride=stride)
                    for _ in range(npaths)]
    
    super().__init__( data_modules, halt_modules, eps=eps )
    
    self.aggregate_batchnorm = nn.BatchNorm2d( self.out_features )
    self.downsample = None
    if stride != 1 or in_features != self.out_features:
      self.downsample = nn.Conv2d( in_features, self.out_features, kernel_size=1, stride=stride, bias=False )
  
  def next_input( self, stage_idx, input, module_output ):
    # All modules receive ACT block input
    return input
  
  def epilog( self, input, output ):
    # Add skip connection and apply final nonlinearities
    if self.downsample is not None:
      input = self.downsample( input )
    out = self.aggregate_batchnorm( output )
    out = out + input # Skip connection
    out = fn.relu( out )
    return out

def resnext_sequential_cifar10( nblocks, widths, npaths, expansion ):
  assert( len(nblocks) == 3 )
  assert( len(widths) == 3 )
  assert( len(npaths) == 3 )
  
  def layer( nblocks, in_features, internal_features, npaths, expansion, downsample ):
    blocks = []
    for i in range(nblocks):
      if i == 0:
        stride = 2 if downsample else 1
        blocks.append( ResNeXtBlock(
          in_features, internal_features, internal_features*expansion, npaths, stride=stride ) )
      else:
        blocks.append( ResNeXtBlock(
          internal_features*expansion, internal_features, internal_features*expansion, npaths ) )
    return blocks
    
  def halt_module( in_features, in_size ):
    return nn.Sequential( 
      nn.AvgPool2d( kernel_size=in_size ),
      FullyConnected( in_features, 1 ),
      nn.Sigmoid() )
  
  image_features = 3
  image_size = (32, 32)
  nclasses = 10
  
  in_features   = [widths[0], widths[0]*expansion, widths[1]*expansion]
  out_features  = [w * expansion for w in widths]
  out_sizes     = [tuple(s // scale for s in image_size) for scale in [1, 2, 4]]
  
  input = nn.Sequential( 
    nn.Conv2d( image_features, widths[0], kernel_size=3, padding=1, bias=False ),
    nn.BatchNorm2d( widths[0] ), nn.ReLU() )
  layers = [
    layer( nblocks[0], in_features[0], widths[0], npaths[0], expansion, downsample=False ),
    layer( nblocks[1], in_features[1], widths[1], npaths[1], expansion, downsample=True ),
    layer( nblocks[2], in_features[2], widths[2], npaths[2], expansion, downsample=True ) ]
  halt_layers = []
  for (nblk, nfeat, size) in zip(nblocks, out_features, out_sizes):
    halt_layers.append( [halt_module( nfeat, size ) for _ in range(nblk)] )
  act_blocks = [SequentialACT( data, halt ) for (data, halt) in zip(layers, halt_layers)]
  avg_pool = nn.AvgPool2d( kernel_size=out_sizes[-1] )
  classify = FullyConnected( out_features[-1], nclasses )
  
  network = [input]
  network.extend( act_blocks )
  network.append( nn.Sequential( avg_pool, classify ) )
  return ACTChainNetwork( network )
