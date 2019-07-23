from   collections import namedtuple
import logging

import torch
from   torch.autograd import Variable

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

PonderCost = namedtuple("PonderCost", ["N", "R"])
PonderCost.__doc__ = """\
Container for the two elements of the "ponder cost":
  `N` : Number of active components
  `R` : "Remainder"
"""

def act_one_shot( h, epsilon=0.01 ):
  """ Computes the ACT gate matrix and ponder cost given the full `BxC` gate
  control matrix `h`, which should contain values in (0, 1). This is the
  "one shot" version that decides how many components to use up-front, rather
  than incrementally.
  
  Parameters:
    `h` : `BxC` matrix of control values in (0, 1)
    `epsilon` : The cutoff value for ACT is `1 - epsilon`. This is needed if
      `h` is the output of a saturating nonlinearity such as logistic.
      
  Returns:
    `(p, rho)` where:
      `p` : `[0,1]^BxC` *real-valued*, *unnormalized* gate matrix containing
        the component weights calculated by ACT.
      `rho` : `PonderCost`
  """
  B = h.size(0)
  # `0` column makes s[N-1] well-defined
  # `1` column indicates "use all components"
  c0 = Variable(torch.zeros(B, 1).type_as(h.data))
  cN = Variable(torch.ones(B, 1).type_as(h.data))
  x = torch.cat( [c0, h, cN], dim=1 )
  s = torch.cumsum( x, dim=1 )
  # Elements that exceed threshold; must convert type to be differentiable
  b = (s >= (1.0 - epsilon)).type_as(h.data)
  # Index of first element that exceeds threshold
  _, N = torch.max( b, dim=1, keepdim=True )
  sN = torch.gather(s, 1, N - 1)
  R = 1.0 - sN # Remainder
  
  p = x.clone()
  # Zero the elements that exceeded the threshold in a differentiable manner
  p = p * (b == 0).type_as(h)
  p.scatter_(1, N, R) # Assign remainders
  # Column 0 is a "guard" column. Column 1 corresponds to using *0*
  # components: If all weight is in Column 1, remaining columns are 0, which
  # is what we want in order to turn everything off. Result is unnormalized.
  p = p[:, 2:]
  # `N` is supposed to be the number of active components, but we want to
  # allow `0` as an option
  N = N - 1
  
  return p, PonderCost(N.type_as(R), R)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
  import torch.nn.functional as fn
  
  h = Variable(torch.rand(4, 3), requires_grad=True)
  x = Variable(torch.rand(4, 3), requires_grad=True)
  print( "h: {}".format(h) )
  print( "x: {}".format(x) )
  p, rho = act_one_shot( h )
  y = p * x
  print( "y: {}".format(y) )
  print( "p: {}".format(p) )
  print( "rho: {}".format(rho) )
  
  L = torch.mean( torch.mean( p, dim=1 ) )
  # L.backward()
  
  w = (rho.N > 1).type_as(x.data)
  # w = torch.max( torch.zeros_like(rho.N), rho.N - 1 ).type_as(x.data)
  Lrho = rho.N + rho.R
  loss = fn.mse_loss(y, x) + Lrho
  loss.backward(w)
  # y.backward(torch.ones_like(y))
  
  print( h.grad )
  print( x.grad )
