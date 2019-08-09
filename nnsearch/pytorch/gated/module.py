import abc
import logging
import math
import pdb
import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fn

from   nnsearch.pytorch.modules import FullyConnected
import nnsearch.pytorch.torchx as torchx

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

class GatedModule(metaclass=abc.ABCMeta):
  @property
  @abc.abstractmethod
  def ncomponents( self ):
    return 0
  
  @property
  @abc.abstractmethod
  def vectorize( self ):
    raise NotImplementedError()
    
  @vectorize.setter
  @abc.abstractmethod
  def vectorize( self, value ):
    raise NotImplementedError()

  @abc.abstractmethod
  def forward( self, x, g ):
    raise NotImplementedError()

# ----------------------------------------------------------------------------

class GatedConcat(GatedModule, nn.Module):
  """ A gated module that concatenates the output of its components. Disabled
  components give an output of 0.
  """
  def __init__( self, components, vectorize=True ):
    super().__init__()
    self.components = nn.ModuleList( components )
    self._vectorize = vectorize

  @property
  def ncomponents( self ):
    return len(self.components)
  
  @property
  def vectorize( self ):
    return self._vectorize
    
  @vectorize.setter
  def vectorize( self, value ):
    self._vectorize = value
  
  def forward( self, x, g ):
    log.debug( "GatedConcat.forward.g: %s", g )
    assert( len(g.size()) == 2 )
    assert( ((g >= 0) + (g <= 1) > 0).all() ) # g \in [0,1]
    if self.vectorize: # GPU-optimized
      out = [c( x ) for c in self.components]
      # Stack components in dim1: [BxNx...] -> BxCxNx...
      # print("COMPONENTS", self.components)
      out = torch.stack( out, dim=1 )
      # gu: BxC -> BxCx1x...
      gu = torchx.unsqueeze_right_as( g, out )
      # print("G",g.shape)
      # print("GU", gu.shape)
      # print("out", out.shape)
      out = gu * out
      out = torch.unbind( out, dim=1 )
      # dim1 is now "channels" (N). out: BxNx...
      out = torch.cat( out, dim=1 )

      # log.micro( "forward.vectorized: %s", out )
    else: # Avoids unnecessary computation
      out = []
      bs = torch.split( x, 1, dim=0 )
      in_shape = x.size()[1:]
      for (i, b) in enumerate(bs):
        Bout = [] # Outputs for this element of batch
        for (j, c) in enumerate(self.components):
          if (g[i,j] > 0).all():
            Cout = g[i,j] * c( b )
            Bout.append( Cout )
          else:
            out_shape = torchx.output_shape( c, input_shape )
            z = Variable(torch.zeros( 1, *out_shape ).type_as(x.data))
            Bout.append( z )
        Bout = torch.cat( Bout, dim=1 ) # Concat channels
        out.append( Bout )
      out = torch.cat( out, dim=0 ) # Concat batch
      log.micro( "forward.gated: %s", out )
    return out

# ----------------------------------------------------------------------------

class GatedSum(GatedModule, nn.Module):
  """ A gated module that sums the outputs of its components. Disabled
  components give an output of 0.
  """
  def __init__( self, components, vectorize=True ):
    super().__init__()
    self.components = nn.ModuleList( components )
    self._vectorize = vectorize

  @property
  def ncomponents( self ):
    return len(self.components)
  
  @property
  def vectorize( self ):
    return self._vectorize
    
  @vectorize.setter
  def vectorize( self, value ):
    self._vectorize = value
  
  def forward( self, x, g ):
    log.debug( "GatedSum.forward.g: %s", g )
    assert( len(g.size()) == 2 )
    assert( ((g >= 0) + (g <= 1) > 0).all() ) # g \in [0,1]
    if self.vectorize: # GPU-optimized
#      out = None
#      for (i, c) in enumerate(self.components):
#        yc = c( x ) #.unsqueeze( 1 )
#        gc = g[:,i]
#        gc = torchx.unsqueeze_right_as( gc, yc )
#        if out is None:
#          out = gc * yc
#        else:
#          out += gc * yc
#        log.info( "GatedSum.forward: out.size(): %s", out.size() )

      out = [c( x ) for c in self.components]
      # Stack components in dim1: [BxNx...] -> BxCxNx...
      out = torch.stack( out, dim=1 )
      
      # gu: BxC -> BxCx1...
      gu = torchx.unsqueeze_right_as( g, out )
      out = gu * out
      # sum over components
      out = torch.sum( out, dim=1 )
      log.micro( "forward.vectorized: %s", out )
    else: # Avoids unnecessary computation
      out = []
      bs = torch.split( x, 1, dim=0 )
      # All components have same shape since we're summing them
      in_shape = x.size()[1:]
      out_shape = torchx.output_shape( self.components[0], in_shape )
      for (i, b) in enumerate(bs):
        Bout = Variable(torch.zeros( 1, *out_shape ).type_as(x.data))
        for (j, c) in enumerate(self.components):
          if (g[i,j] > 0).all():
            Cout = g[i,j] * c( b )
            Bout += Cout
        out.append( Bout )
      out = torch.cat( out, dim=0 ) # Concat batch
      log.micro( "forward.gated: %s", out )
    return out

# ----------------------------------------------------------------------------

class BlockGatedConv1d(GatedConcat):
  def __init__( self, ncomponents, in_channels, out_channels, kernel_size, **kwargs ):
    """
    Parameters:
      `ncomponents` : Number of gated blocks. Must divide `out_channels`.
    """
    assert( out_channels % ncomponents == 0 )
    out_channels //= ncomponents
    components = [nn.Conv1d( in_channels, out_channels, kernel_size, **kwargs )
                  for _ in range(ncomponents)]
    super().__init__( components )
    
class BlockGatedConv2d(GatedConcat):
  def __init__( self, ncomponents, in_channels, out_channels, kernel_size, **kwargs ):
    """
    Parameters:
      `ncomponents` : Number of gated blocks. Must divide `out_channels`.
    """
    assert( out_channels % ncomponents == 0 )
    out_channels //= ncomponents
    components = [nn.Conv2d( in_channels, out_channels, kernel_size, **kwargs )
                  for _ in range(ncomponents)]
    super().__init__( components )
    
class BlockGatedConv3d(GatedConcat):
  def __init__( self, ncomponents, in_channels, out_channels, kernel_size, **kwargs ):
    """
    Parameters:
      `ncomponents` : Number of gated blocks. Must divide `out_channels`.
    """
    assert( out_channels % ncomponents == 0 )
    out_channels //= ncomponents
    components = [nn.Conv3d( in_channels, out_channels, kernel_size, **kwargs )
                  for _ in range(ncomponents)]
    super().__init__( components )
    
class BlockGatedFullyConnected(GatedConcat):
  """ A fully-connected layer. Flattens its input automatically.
  """
  def __init__( self, ncomponents, in_channels, out_channels, **kwargs ):
    assert( out_channels % ncomponents == 0 )
    out_channels //= ncomponents
    components = [FullyConnected( in_channels, out_channels, **kwargs )
                  for _ in range(ncomponents)]
    super().__init__( components )
    
# ----------------------------------------------------------------------------

class GatedModuleList(nn.Module):
  def __init__( self, gate, modules ):
    super().__init__()
    self.gate = gate
    self.fn = nn.ModuleList( modules )
    
  def forward( self, x ):
    self.gate.reset()
    for m in self.fn:
      if issubclass(m.__class__, GatedModule):
        self.gate.next_module( m )
        g = self.gate( x )
        log.debug( "m.g: %s\n%s", m, g )
        x = m( x, g )
      else:
        x = m( x )
    return x
    
# ----------------------------------------------------------------------------

class GatedChainNetwork(nn.Module):
  def __init__( self, gate, modules, gated_modules,
                normalize=True, reverse_normalize=False, gbar_info=False ):
    """
    Parameters:
    -----------
      `gate`: Gate policy for whole network
      `modules`: List of all modules in the network
      `gated_modules`: List of GatedModules in the network
      `reverse_normalize`: If `True`, scale gate values *up* so that the sum of
        the gate vectors is equal to the number of components. If `False`,
        scale gate values *down* so that the sum is equal to 1.
      `gbar_info`: Log extra information about the gbar matrices.
    """
    super().__init__()
    self.gate = gate
    self.fn = nn.ModuleList( modules )
    self._gated_modules = gated_modules
    self.normalize = normalize
    self.reverse_normalize = reverse_normalize
    self._gbar_info = gbar_info
    
  @property
  def gated_modules( self ):
    yield from self._gated_modules
    
  def _normalize( self, g ):
    z = 1e-12 + torch.sum(g, dim=1, keepdim=True) # Avoid divide-by-zero
    if self.reverse_normalize:
      return g * g.size(1) / z
    else:
      return g / z
  
  def _log_gbar( self, gbar ):
    log.debug( "network.gbar:\n%s", gbar )
    if self._gbar_info:
      N = torch.sum( (gbar.data > 0).float(), dim=1, keepdim=True )
      # log.info( "network.log_gbar.N:\n%s", N )
      g = N * gbar.data
      # log.info( "network.log_gbar.g:\n%s", g )
      h = torchx.histogram( g,
        bins=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10] )
      log.info( "network.g.hist:\n%s", torchx.pretty_histogram( h ) )
  
  def set_gate_control( self, u ):
    self._u = u
  
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
        # print(m, g)
        x = m( x, g )
      else:
        x = m( x )
      # print(x.size())
    log.debug( "network.x: %s", x )
    return x, gs
