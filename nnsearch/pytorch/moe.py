import logging
import math

import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as fn

log = logging.getLogger( __name__ )

# FIXME: Based on experience with adaptive.MonotonicGatedChannelStack, probably
# needs to be changed to work with experts other than nn.Linear
# FIXME: Need custom autograd function to get gradients to propagate to gate layers
class SparselyGatedMixtureOfExperts(nn.Module):
  """ Implements the Sparsely-Gated Mixture-of-Experts layer. This version is a
  direct implementation of the mathematical formulation, and may be inefficient.
  
  @article{shazeer2017outrageously,
    title={Outrageously large neural networks: The sparsely-gated mixture-of-experts layer},
    author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, 
      Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff},
    journal={arXiv preprint arXiv:1701.06538},
    year={2017}
  }
  """
  def __init__( self, experts, Wg, Wnoise, top_k, ):
    super().__init__()
    self.nexperts = len(experts)
    self.experts = experts
    for (i, e) in enumerate(experts):
      self.add_module( "e{}".format(i), e )
    self.Wg = Wg
    self.Wnoise = Wnoise
    self.top_k = top_k
    
  def forward( self, x ):
    """ Implements Eq. (1, 3-5) in the paper.
    """
    # FIXME: Bug in PyTorch (https://github.com/pytorch/pytorch/issues/3397)
    # prevents using infinity here -> use very big value
    Inf = 1e38 # Close to maximum float32 value
    # -- Eq. 4
    g = self.Wg(x)
    noise = ag.Variable(torch.randn(self.nexperts)) * fn.softplus(self.Wnoise(x))
    H = g + noise
    # -- Eq. 5
    # Assign -Inf to the (N - k) *smallest* indices
    _val, idx = torch.topk( H, largest=False, k=(self.nexperts - self.top_k) )
    # Note: Non-inplace `scatter` is undocumented in PyTorch 0.3.1
    keep = H.scatter( dim=1, index=idx, source=-Inf )
    # -- Eq. 3
    G = fn.softmax( keep, dim=1 )
    log.debug( "sgmoe.G: %s", G )
    
    # Note: We still evaluate e(x) even if G(x) == 0 (c.f. DynamicSGMoE)
    E = torch.stack( [e(x) for e in self.experts], dim=1 )
    # Append singleton dimensions to G until is has same number as E
    while len(G.size()) < len(E.size()):
      G = G.unsqueeze(dim=len(G.size()))
    # Expand gate values along extra dimensions
    G = G.expand_as( E )
    # Element-wise multiply by expanded gate function
    output = G * E
    # Sum of expert outputs
    return torch.sum( output, dim=1 )

# FIXME: Based on experience with adaptive.MonotonicGatedChannelStack, probably
# needs to be changed to work with experts other than nn.Linear
# FIXME: Need custom autograd function to get gradients to propagate to gate layers
class DynamicSparselyGatedMixtureOfExperts(nn.Module):
  """ Implements the Sparsely-Gated Mixture-of-Experts layer. This version
  avoids evaluating experts that are not selected by the gating function.
  
  @article{shazeer2017outrageously,
    title={Outrageously large neural networks: The sparsely-gated mixture-of-experts layer},
    author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, 
      Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff},
    journal={arXiv preprint arXiv:1701.06538},
    year={2017}
  }
  """
  def __init__( self, experts, Wg, Wnoise, top_k ):
    super().__init__()
    self.nexperts = len(experts)
    self.experts = experts
    for (i, e) in enumerate(experts):
      self.add_module( "e{}".format(i), e )
    self.Wg = Wg
    self.Wnoise = Wnoise
    self.top_k = top_k
    
  def forward( self, x ):
    """ Implements Eq. (1, 3-5) in the paper.
    """
    # FIXME: Bug in PyTorch (https://github.com/pytorch/pytorch/issues/3397)
    # prevents using infinity here -> use very big value
    Inf = 1e38 # Close to maximum float32 value
    # -- Eq. 4
    g = self.Wg(x)
    noise = ag.Variable(torch.randn(self.nexperts)) * fn.softplus(self.Wnoise(x))
    H = g + noise
    # -- Eq. 5
    # Assign -Inf to the (N - k) *smallest* indices
    _val, idx = torch.topk( H, largest=False, k=(self.nexperts - self.top_k) )
    # Note: Non-inplace `scatter` is undocumented in PyTorch 0.3.1
    keep = H.scatter( dim=1, index=idx, source=-Inf )
    # -- Eq. 3
    G = fn.softmax( keep, dim=1 )
    log.debug( "sgmoe.G: %s", G )
    
    # Evaluate selected experts
    Ebatch = []
    for (xb, G_row, I_row) in zip(x, G, idx): # each sample in batch
      Erow = None
      iset = set( i.data[0] for i in I_row )
      for j in range(self.nexperts):
        if j not in iset:
          Ej = G_row[j] * self.experts[j](xb)
          if Erow is None:
            Erow = Ej
          else:
            Erow += Ej
      Ebatch.append( Erow )
    # TODO: I'm guessing that making a list and stacking it all at once is
    # more efficient than incremental stacking. Should profile this on GPU.
    Ebatch = torch.stack( Ebatch, dim=0 )
    return Ebatch

if __name__ == "__main__":
  # Verify same output from both implementations
  logging.basicConfig()
  logging.getLogger().setLevel( logging.DEBUG )

  # 42 = Expert 1 never used
  # 48 = All experts used and not used
  torch.manual_seed( 48 )
  input = ag.Variable(torch.randn( 4, 3 ))
  print( input )
  fake_loss = torch.randn( 4, 2 )
  experts = [nn.Linear(3, 2), nn.Linear(3, 2), nn.Linear(3, 2)]
  Wg = nn.Linear( 3, len(experts) )
  Wnoise = nn.Linear( 3, len(experts) )
  
  smix = SparselyGatedMixtureOfExperts( experts, Wg, Wnoise, 2 )
  dmix = DynamicSparselyGatedMixtureOfExperts( experts, Wg, Wnoise, 2 )
  
  print( "==== Static" )
  torch.manual_seed( 314 )
  output = smix(input)
  print( output )
  output.backward( fake_loss )
  for e in smix.experts:
    print( e.weight.grad.data )
  for p in smix.parameters():
    if p.grad is not None:
      p.grad.data.zero_()
  
  print( "==== Dynamic" )
  torch.manual_seed( 314 )
  output = dmix(input)
  print( output )
  output.backward( fake_loss )
  for e in dmix.experts:
    print( e.weight.grad.data )
  for p in dmix.parameters():
    if p.grad is not None:
      p.grad.data.zero_()
