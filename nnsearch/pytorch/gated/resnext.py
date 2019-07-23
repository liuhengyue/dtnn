from   collections import namedtuple
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as fn

import nnsearch.pytorch.data as data
from   nnsearch.pytorch.gated.module import GatedChainNetwork, GatedModule, GatedSum
from   nnsearch.pytorch.models.resnext import ResNeXtBlock
from   nnsearch.pytorch.modules import FullyConnected
import nnsearch.pytorch.torchx as torchx

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

ResNeXtStage = namedtuple( "ResNeXtStage",
  ["nlayers", "nchannels", "ncomponents", "expansion"] )

class GatedResNeXtBlock(GatedSum, nn.Module):
  def __init__( self, in_channels, internal_channels, out_channels,
                ncomponents, stride=1, 
                skip_connection_batchnorm=True, vectorize=True ):
    """
    ResNeXt "basic block", divided into separately-gated "paths" (Figure 3a
    in the paper).
    
    Parameters:
      `in_channels`: Number of input channels.
      `internal_channels`: Number of output channels for the first two layers.
      `out_channels`: Number of output channels for the block.
      `ncomponents`: Number of separately-gated paths in the block. Must be a
        factor of `internal_channels`. This is called the "cardinality" in the
        paper.
      `stride`: If > 1, the first layer in the block performs convolutional
        down-sampling.
      `skip_connection_batchnorm`: If False, don't apply batchnorm to the output of
        the skip down-sampling layer before adding it to the residual output.
        `skip_connection_batchnorm = False` is the *legacy* behavior, but not
        intentionally because it is not faithful to the ResNeXt paper; thus it
        is no longer the default.
      `vectorize`: Vectorize the computation for speed on GPU
    """
    # FIXME: Don't see any technical reason to enforce this
    assert( internal_channels % ncomponents == 0 )
    path_channels = internal_channels // ncomponents
    
    def path():
      return nn.Sequential(
        nn.Conv2d( in_channels, path_channels, kernel_size=1, bias=False ),
        nn.BatchNorm2d( path_channels ), nn.ReLU(),
        nn.Conv2d( path_channels, path_channels, kernel_size=3, stride=stride,
          padding=1, bias=False ),
        nn.BatchNorm2d( path_channels ), nn.ReLU(),
        nn.Conv2d( path_channels, out_channels, kernel_size=1, bias=False ) )
        # No BatchNorm / ReLU after last conv
    
    components = [path() for _ in range(ncomponents)]
    super().__init__( components, vectorize=vectorize )
    
    self.aggregate_batchnorm = nn.BatchNorm2d( out_channels )
    self.downsample = None
    if stride != 1 or in_channels != out_channels:
      if skip_connection_batchnorm:
        self.downsample = nn.Sequential(
          nn.Conv2d( in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False ),
          nn.BatchNorm2d( out_channels ) )
      else: # Legacy behavior
        self.downsample = nn.Conv2d( in_channels, out_channels,
          kernel_size=1, stride=stride, bias=False )
  
  def forward( self, x, g ):
    skip = x
    if self.downsample is not None:
      skip = self.downsample( skip )
      
    out = super().forward( x, g )
    out = self.aggregate_batchnorm( out )
    out = out + skip
    out = fn.relu( out )
    return out
    
# ----------------------------------------------------------------------------

class GatedGroupedResNeXtBlock(ResNeXtBlock, GatedModule):
  def __init__( self, in_channels, internal_channels, out_channels,
                ncomponents, stride=1, skip_connection_batchnorm=True ):
    """
    Gated version of ResNext "C" block.
    
    Parameters:
      `in_channels`: Number of input channels.
      `internal_channels`: Number of output channels for the first two layers.
      `out_channels`: Number of output channels for the block.
      `ncomponents`: Number of separately-gated paths in the block. Must be a
        factor of `internal_channels`. This is called the "cardinality" in the
        paper.
      `stride`: If > 1, the first layer in the block performs convolutional
        down-sampling.
    """
    # This version does not implement the option to use the (incorrect) legacy
    # behavior, but we need to be able to pass the parameter for genericity
    if not skip_connection_batchnorm:
      raise NotImplementedError( "skip_connection_batchnorm == False" )
    super().__init__(
      in_channels, internal_channels, out_channels, ncomponents, stride )
    self._group_size = internal_channels // ncomponents
  
  @property
  def ncomponents( self ):
    return self.ngroups
  
  @property
  def vectorize( self ):
    return True
    
  @vectorize.setter
  def vectorize( self, value ):
    raise NotImplementedError()
  
  def forward( self, x, g ):
    skip = self.skip( x )
    
    # 'g' is size BxC with one entry for each group; expand it so size BxN by 
    # repeating the gate value for each member of the group 
    assert len(g.size()) == 2 and g.size(1) == self.ncomponents
    exsize = list(g.size()) + [self._group_size]
    gex = g.unsqueeze(2).expand(*exsize).contiguous().view(g.size(0), -1)
    assert gex.size(1) == (self.ncomponents * self._group_size)
    log.debug( "gex:\n%s", gex )
    gex = torchx.unsqueeze_right_as( gex, x )
    
    for (i, m) in enumerate(self.residual._modules.values()):
      x = m( x )
      # Finished 2nd conv; apply gating.
      # Note: BatchNorm behavior will be different from architecture A, since
      # in A the data never reaches the BatchNorm if the path is gated off. 
      if i == 5: 
        # log.debug( "x:\n%s", torch.sum( torch.sum( x[:,0,:,:], dim=2 ), dim=1 ) )
        x *= gex
        # log.debug( "xg:\n%s", torch.sum( torch.sum( x[:,0,:,:], dim=2 ), dim=1 ) )
      
    out = x + skip
    out = fn.relu( out )
    return out

# ----------------------------------------------------------------------------

ResNeXtDefaults = namedtuple("ResNeXtDefaults", ["input", "in_shape", "expansion"])

def defaults( dataset ):
  if isinstance(dataset, (data.Cifar10Dataset, data.SvhnDataset)):
    in_channels = 64
    input = nn.Sequential( 
      nn.Conv2d( 3, in_channels, 3, padding=1, bias=False ),
      nn.BatchNorm2d( 64 ), nn.ReLU() )
    in_shape = (in_channels, 32, 32)
    expansion = 4
  elif isinstance(dataset, data.ImageNetDataset):
    in_channels = 64
    input = nn.Sequential(
      nn.Conv2d( dataset.in_shape[0], in_channels, 7,
                 stride=2, padding=3, bias=False ),
      nn.BatchNorm2d( in_channels ),
      nn.ReLU(),
      nn.MaxPool2d( 3, stride=2, padding=1 ) )
    in_shape = torchx.output_shape( input, dataset.in_shape )
    expansion = 2 # Note: ResNet expansion is 4, but ResNeXt is 2
  else:
    raise NotImplementedError()
  return ResNeXtDefaults(input, in_shape, expansion)

# ----------------------------------------------------------------------------

class GatedResNeXt(GatedChainNetwork):
  def __init__( self, gate, input, in_shape, nclasses, stages, *,
                resnext_block_t=GatedResNeXtBlock, 
                skip_connection_batchnorm=True, **kwargs ):
    gated_modules = []
    modules = [input]
    def make_stage( nlayers, ncomponents, in_shape, in_channels,
                    internal_channels, out_channels, downsample_stride=1 ):
      layers = []
      for i in range(nlayers):
        if i == 0:
          m = resnext_block_t( in_channels, internal_channels, 
            out_channels, ncomponents, stride=downsample_stride,
            skip_connection_batchnorm=skip_connection_batchnorm )
          layers.append( m )
          shape = tuple([in_channels]) + in_shape[1:]
          gated_modules.append( (m, shape) )
        else:
          m = resnext_block_t( out_channels,
            internal_channels, out_channels, ncomponents,
            skip_connection_batchnorm=skip_connection_batchnorm )
          layers.append( m )
          shape = tuple([out_channels]) + in_shape[1:]
          gated_modules.append( (m, shape) )
      return layers
    
    in_channels = in_shape[0]
    for (i, stage) in enumerate(stages):
      out_channels = stage.nchannels * stage.expansion
      downsample_stride = 2 if i > 0 else 1
      modules.extend( make_stage( stage.nlayers, stage.ncomponents,
        in_shape, in_channels, stage.nchannels, out_channels,
        downsample_stride=downsample_stride ) )
      in_shape = tuple(
        [out_channels] + [s // downsample_stride for s in in_shape[1:]] )
      in_channels = out_channels
    # Global average pool
    modules.append( nn.AvgPool2d( kernel_size=in_shape[1:] ) )
    # Classifier
    modules.append( FullyConnected( in_channels, nclasses ) )
    super().__init__( gate, modules, gated_modules, **kwargs )
