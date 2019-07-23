import logging

from torch import nn
from torch.nn import functional as fn

from nnsearch.pytorch import torchx
from nnsearch.pytorch.gated.module import GatedChainNetwork, GatedModule
from nnsearch.pytorch.models import resnet

log = logging.getLogger( __name__ )

class BlockDropResNetBlock(resnet.ResNetBlock, GatedModule):
  """ A ResNet block where the entire residual module can be bypassed.
  """
  @property
  def ncomponents( self ):
    return 1
  
  @property
  def vectorize( self ):
    return True
    
  @vectorize.setter
  def vectorize( self, value ):
    if not value:
      raise ValueError( "does not support vectorize = False" )
  
  def forward( self, x, g ):
    # TODO: Since `g` is one-dimensional for blockdrop, it might be possible
    # to speed things up by grouping all the active components together, 
    # computing the residual on only those, and them putting the result back
    # in its original order.
    # See `epsilon.rl.multitask.pytorch.dqn.TaskIndexedDqn` for an example.
    skip = self.skip( x )
    log.debug( "blockdrop.g:\n%s", g )
    y = torchx.unsqueeze_right_as(g, x) * self.residual( x )
    log.nano( "blockdrop: g*f:\n%s", y )
    out = skip + y
    out = self.activation( out )
    return out
    
class BlockDropResNetBottleneckBlock(resnet.ResNetBottleneckBlock, GatedModule):
  """ A ResNet block where the entire residual module can be bypassed.
  """
  @property
  def ncomponents( self ):
    return 1
  
  @property
  def vectorize( self ):
    return True
    
  @vectorize.setter
  def vectorize( self, value ):
    if not value:
      raise ValueError( "does not support vectorize = False" )
  
  def forward( self, x, g ):
    # TODO: Since `g` is one-dimensional for blockdrop, it might be possible
    # to speed things up by grouping all the active components together, 
    # computing the residual on only those, and them putting the result back
    # in its original order.
    # See `epsilon.rl.multitask.pytorch.dqn.TaskIndexedDqn` for an example.
    log.debug( "blockdrop.g:\n%s", g )
    skip = self.skip( x )
    y = torchx.unsqueeze_right_as(g, x) * self.residual( x )
    log.nano( "blockdrop: g*f:\n%s", y )
    out = skip + y
    out = self.activation( out )
    return out

class BlockDropResNetStage(nn.Sequential, GatedModule):
  def __init__( self, block_t, nblocks, in_channels, internal_channels, stride=1 ):
    blocks = []
    expansion = 4 if issubclass(block_t, BlockDropResNetBottleneckBlock) else 1
    for i in range(nblocks):
      if i == 0:
        blocks.append( block_t( in_channels, internal_channels, stride=stride ))
      else:
        blocks.append( block_t( expansion*internal_channels, internal_channels ))
    # for i in range(nblocks):
      # if i == 0:
        # blocks.append( block_t( in_channels, out_channels, stride=stride ))
      # else:
        # blocks.append( block_t( out_channels, out_channels ))
    super().__init__( *blocks )
    
  @property
  def ncomponents( self ):
    return 1
  
  @property
  def vectorize( self ):
    return True
    
  @vectorize.setter
  def vectorize( self, value ):
    if not value:
      raise ValueError( "does not support vectorize = False" )
  
  def forward( self, x, g ):
    for (i, m) in enumerate(self):
      x = m( x, g[:,i].contiguous() )
    return x
    
class BlockDropResNet(GatedChainNetwork):
  def __init__( self, gate, block_t, input, in_shape, stages, output, **kwargs ):
    """ Creates a ResNet network that follows the architectural pattern
    from the paper. One detail to keep in mind that is not immediately
    obvious is that the first ResNet stage does *not* perform subsampling.
    
    Parameters:
    -----------
      `gate`: Gate controller
      `input`: An arbitrary input module
      `in_shape`: The shape of the output of the input module
      `stages`: List of ResNetStageSpec tuples
      `output`: An output module to apply after the chain of ResNet stages. For
        a standard ResNet classifier, this would be GlobalAvgPool2d followed
        by FullyConnected.
      `kwargs`: Additional keyword arguments for `GatedChainNetwork`.
    """
    modules = [input]
    gated_modules = []
    in_channels = in_shape[0]
    
    for (i, stage) in enumerate(stages):
      out_channels = stage.nchannels
      downsample_stride = 2 if i > 0 else 1
      m = BlockDropResNetStage(
        block_t, stage.nblocks, in_channels, out_channels, stride=downsample_stride )
      modules.append( m )
      in_shape = torchx.output_shape( m, in_shape )
      in_channels = in_shape[0]
      gated_modules.append( (m, in_shape) )
    
    modules.append( output )
    
    super().__init__( gate, modules, gated_modules, normalize=False, **kwargs )
