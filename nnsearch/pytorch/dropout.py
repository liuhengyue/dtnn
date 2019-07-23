import logging
import math

import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as fn

log = logging.getLogger( __name__ )

class BlockDropout(nn.Module):
  """ Implements uniform random dropout at the "block" level -- instead of
  dropping out individual units, drops out sets of units.
  
  The components can be any Module instance (layers up to entire networks), as
  long as their inputs and outputs have the same shape (except for number of 
  output channels).
  """
  
  def __init__( self, in_features, components, out_shapes, drop_rate, normalize_output=True ):
    """
    Parameters:
      `in_features` : Number of input features
      `components` : List of component modules
      `out_shapes` : List of shapes of component outputs
      `Wg` : Module for learning the gating function
      `initial_slope` : Starting value of slope parameter for slope-annealed
        Bernoulli gradient estimator
    """
    super().__init__()
    self.ncomponents = len(components)
    self.components  = components
    self.in_features = in_features
    self.out_shapes = out_shapes
    self.drop_rate = drop_rate
    self.normalize_output = normalize_output
    for (i, c) in enumerate(components):
      self.add_module( "c{}".format(i), c )
  
  def forward( self, x ):
    # Evaluate selected experts
    Ebatch = []
    batch_size = x.size(0)
    G = torch.zeros( batch_size, self.ncomponents ).type_as(x.data)
    G.bernoulli_( 1.0 - self.drop_rate )
    log.verbose( "dropout.bernoulli: %s", G )
    always_on = torch.multinomial( torch.ones_like(G), 1 )
    log.verbose( "dropout.always_on: %s", always_on )
    G.scatter_( 1, always_on, 1 ) # Ensure at least one block is turned on
    log.verbose( "dropout.G: %s", G )
    for (b, xb) in enumerate(x): # each sample in batch
      xb = xb.unsqueeze( dim=0 )
      Yb = []
      active_features = 0
      for i in range(self.ncomponents):
        if G[b,i] > 0:
          yi = self.components[i](xb)
          if self.normalize_output:
            Ci = yi.size(1)  # Number of output features from this component
            active_features += Ci
            Yb.append( Ci * yi ) # Weight the output for later normalization
          else:
            Yb.append( yi )
        else:
          # Output 0's
          pad = torch.zeros( 1, *self.out_shapes[i] ).type_as(x.data)
          Yb.append( ag.Variable(pad) )
      Yb = torch.cat( Yb, dim=1 )
      if self.normalize_output and active_features > 0:
        Yb /= active_features # Normalize
      log.verbose( "gate.Gb: %s %s", b, G[b] )
      log.verbose( "gate.Yb: %s %s", b, Yb )
      Ebatch.append( Yb )
    # TODO: I'm guessing that making a list and stacking it all at once is
    # more efficient than incremental stacking. Should profile this on GPU.
    Ebatch = torch.cat( Ebatch, dim=0 )
    return Ebatch
