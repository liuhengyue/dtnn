import torch
from torch.autograd import Variable
import torch.nn as nn

import nnsearch.pytorch.gated.relax as relax

class TestModule(nn.Module):
  def forward( self, x ):
    B = x.size(0)
    C = x.size(1)
    t = relax.binary_straight_through.apply( x )
    print( t )
    c = torch.sum( t, dim=1, keepdim=True )
    print( c )
    g = Variable(torch.arange(1, C+1).expand(B, C))    # Each row is 1,...,n
    p = c.expand(B, C) # Each column == c
    g = (g <= p).type_as(c) # Convert to [1 1 ... 1 0 0 ... 0] numeric
    z = c + 1e-12
    return g / z

def gate_matrix_from_count( c, n ):
  print( "c:{}".format(c) )
  batch_size = c.size(0)
  idx = Variable(torch.arange(1, n+1).expand(batch_size, n)) # Each row == [1,...,n]
  g = c.unsqueeze(-1).expand(batch_size, n).clone()      # Each column == c
  g[g.detach() < idx] = 0
  z = torch.sum( g, dim=1, keepdim=True ) + 1e-12
  g /= z
  return g

if __name__ == "__main__":
  B = 4
  C = 3
  x = Variable( torch.randn( B, C ), requires_grad=True )
  print( "x: {}".format(x) )
  
  # m = TestModule()
  # y = m(x)
  # print( y )
  # y.backward( torch.ones_like(y) )
  # print( x.grad )
  
  # t = relax.binary_straight_through.apply( x )
  p = 0.5 * torch.ones( 4, 3 )
  t = Variable(torch.bernoulli( p ), requires_grad=True)
  print( "t: {}".format(t) )
  c = torch.sum( t, dim=1 )
  # c = Variable(torch.Tensor( [0, 1, 2, 3] ), requires_grad=True)
  print( "c: {}".format(c) )
  g = relax.gate_matrix_from_count.apply( c, C )
  # g = gate_matrix_from_count( c, C )
  print( "g: {}".format(g) )
  y = x * g
  print( "y: {}".format(y) )
  # y.backward( torch.ones_like(y) )
  y.backward( torch.rand(B, C) )
  print( x.grad )
  print( t.grad )
  print( c.grad )
