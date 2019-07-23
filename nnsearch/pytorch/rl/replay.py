import abc
from   collections.abc import Sequence
import heapq
import math

# ----------------------------------------------------------------------------

class ReplayMemory(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def offer( self, x ):
    """ Possibly add `x` to the replay memory at implementation's discretion.
    """
    pass
    
  @abc.abstractmethod
  def sample( self, rng, batch_size ):
    """ Returns a list of "ids" that identify objects in the memory, sampled
    at random using `rng` for random numbers.
    
    Client code should treat the returned "ids" as opaque objects, and should
    not rely on any of their properties.
    """
    pass
    
  @abc.abstractmethod
  def get( self, ids ):
    """ Returns an iterator over the elements identified by `ids`, in the same
    order.
    """
    pass
    
class PrioritizedReplayMemory(ReplayMemory):
  @abc.abstractmethod
  def priority( self, i ):
    """ Returns the scalar priority of the object identified by `i`.
    
    Parameters:
      `i` : Object identifier
    
    Returns:
      Priority values
    """
    pass
    
  @abc.abstractmethod
  def update( self, i, new_priority ):
    """ Set the priority of object `i` to `new_priority`.
    """
    pass

# ----------------------------------------------------------------------------

class CircularList(ReplayMemory, Sequence):
  """ A list in which insertions overwrite the oldest element once the list
  reaches a maximum size.
  """
  def __init__( self, N ):
    self._data = []
    self._N = N
    self._i = 0
    
  def offer( self, x ):
    if len(self._data) < self._N:
      self._data.append( x )
    else:
      self._data[self._i] = x
    self._i = (self._i + 1) % self._N
    
  def sample( self, rng, k ):
    return [rng.randrange(len(self._data)) for _ in range(k)]
    
  def get( self, ids ):
    for i in ids:
      yield self._data[i]
    
  def __iter__( self ):
    for i in len(self):
      yield self._data[i]
    
  def __getitem__( self, i ):
    return self._data[i]
    
  def __len__( self ):
    return len(self._data)
    
# ----------------------------------------------------------------------------

def _harmonic_spacings( N, k ):
  """ Returns a list `idx` of `k` indices such that
  `H(idx[0]) \approx H(idx[i]) - H(idx[i-1]) \approx C` for some constant `C`,
  where `H(n)` is the `n`th harmonic number. The indices cover the entire range
  of `N`, in the sense that `idx[-1] == N`
  """
  c = math.pow(N, 1.0/k)
  h = c
  # The choice to use `int(round(h)) - 1` gives empirically smaller MSE than
  # e.g. using math.ceil and/or not subtracting 1. The error tends to
  # concentrate in the first interval, which is acceptable for our purposes
  # because that interval should contain the most-informative examples.
  idx = [max( 1, int(round(h)) - 1 )]
  for i in range(1, k - 1):
    h *= c
    j = int(round(h)) - 1
    idx.append( max(j, idx[i-1] + 1) )
  idx.append( N )
  return idx
  
def harmonic_number( k ):
  gamma = 0.57721566490153 # Euler-Mascheroni constant
  if k < 100:
    # Approximation is poor for small values
    return sum( 1/i for i in range(1, k+1) )
  else:
    return math.log(k) + gamma
    
def harmonic_spacings( N, k, alpha=1 ):
  """ Returns a a tuple `(idx, Z)` such that for `m = idx[i]`, `n = idx[i+1]`,
  `sum( math.pow(1/j, alpha) for j in range(m, n) ) \approx Z / k` and
  `idx[-1] == N`.
  
  Parameters:
    `N` : Length of list
    `k` : Number of parts
    `alpha` : "Smoothing" exponent
    
  Returns:
    `(idx, Z)` : `idx` is a list of interval endpoints, `Z` is the normalizing
      factor for converting the smoothed harmonic series to a probability.
  """
  # Could approximate this with the identity \int_1^a x^n dx = (a^{n+1} - 1)/(n+1)
  # See: https://en.wikipedia.org/wiki/Cavalieri%27s_quadrature_formula
  H = [0] * N
  h = 0
  for i in range(N):
    if alpha == 1:
      h += 1 / (i+1)
    else:
      h += 1 / math.pow( i+1, alpha )
    H[i] = h
  Z = H[-1]
  I = Z / k
  idx = []
  s = 0
  for i in range(N):
    if H[i] - s > I:
      idx.append( i+1 )
      s += I
  if len(idx) < k:
    idx.append( N )
  assert( len(idx) == k )
  return (idx, Z)

# FIXME: Update to use the new PrioritizeReplayMemory ABC
class RankPrioritizedReplay:
  """ Samples approximately proportional to the inverse of the sample's rank.
  """
  def __init__( self, N, k, alpha=1, beta=1 ):
    """
    Parameters:
      `N` : Maximum size of the replay buffer
      `k` : Number of "blocks" for stratified sampling
      `alpha` : Smoothing parameter for item probability
      `beta` : Smoothing parameter for importance weights
    """
    self.N = N
    self.k = k
    self._alpha = alpha
    self.beta = beta
    self._data = []
    self._last_size = len(self._data)
    self._max_p = 0
    self._boundaries = []
    self._Z = None
    self._delete = (self.N // 2) + 1
  
  @property
  def prioritized( self ):
    return True
  
  @property
  def alpha( self ):
    return self._alpha
    
  @alpha.setter
  def alpha( self, new_alpha ):
    if self._alpha != new_alpha:
      self._alpha = new_alpha
      self._calculate_boundaries()
  
  @property
  def size( self ):
    return len(self._data)
  
  @property
  def full( self ):
    return self.size == self.N
  
  def _convert_value( self, p ):
    return -p
  
  def _swap( self, i, j ):
    tmp = self._data[i]
    self._data[i] = self._data[j]
    self._data[j] = tmp
    
  def _greater( self, i, j ):
    return self._data[i][0] > self._data[j][0]
  
  def _parent( self, j ):
    return (j - 1) // 2
  
  def _children( self, j ):
    """ Returns a tuple of length 0, 1, or 2. If length == 2, the children are
    sorted by key, then by index.
    """
    c1 = 2*j + 1
    c2 = c1 + 1
    if c1 >= len(self._data):
      return ()
    if c2 >= len(self._data):
      return (c1,)
    if self._greater(c1, c2):
      return (c2, c1)
    return (c1, c2)
    
  def add( self, x ):
    """ Add a new example, defaulting to the max priority (front of queue).
    """
    qsup = self._convert_value(self._max_p + 1) # Inflated priority
    if len(self._data) == self.N: # Move a low-priority element to front
      mid = len(self._data) // 2 - 1
      d = self._children(mid)[-1]
      self._data[d] = (qsup, x)
      self._reprioritize( d, qsup )
    else:
      heapq.heappush( self._data, (qsup, x) ) # Insert at front
    self._data[0] = (self._convert_value(self._max_p), x) # Correct priority
    return 0
  
  def _reprioritize( self, i, q ):
    """ Change the priority of the item currently at index `i` to `q`, where
    `q` is a min-heap priority.
    """
    def up( i ):
      while i > 0:
        pi = self._parent(i)
        if self._greater(pi, i):
          self._swap(pi, i)
          i = pi
        else:
          break
    def down( i ):
      while True:
        cs = self._children(i)
        if len(cs) == 0:
          break
        least_child = cs[0]
        if not self._greater(i, least_child):
          break
        self._swap(i, least_child)
        i = least_child
    
    (old_q, e) = self._data[i]
    self._data[i] = (q, e)
    up( i )
    down( i )

  def update( self, i, new_p ):
    """ Change the priority of the item at index `i` to `new_p`.
    """
    new_q = self._convert_value( new_p )
    self._reprioritize( i, new_q )
    self._max_p = self._convert_value( self._data[0][0] )
  
  def sample( self, rng, k ):
    """ Sample `k` replays according to `probability()`.
    
    Parameters:
      `rng`
      `k` : Number of samples
      
    Returns:
      `(samples, idx, weights)` : Three lists of length `k`
    """
    blocks = []
    N = min(len(self._data), self.k)
    for _ in range(k // N): # Round-robin block sampling to reduce variance
      blocks.extend( range(1, N+1) )
    for _ in range(k % N): # Choose blocks at random for the remainder
      blocks.append( rng.randrange(1, N+1) )
    return self.sample_from_blocks( rng, blocks )
  
  def sample_from_blocks( self, rng, blocks ):
    """ Sample one replay uniformly at random from each block index
    `{0,...,k-1}` in `blocks`.
    
    Parameters:
      `rng`
      `blocks` : Block indices
      
    Returns:
      `(samples, idx, weights)` : Three lists of length `len(blocks)`
    """
    if len(self._data) != self._last_size:
      self._calculate_boundaries()
      self._last_size = len(self._data)
    samples = []
    idx     = []
    weights = []
    max_w = self.max_importance()
    for b in blocks:
      start = self._boundaries[b - 1]
      end   = self._boundaries[b]
      i = rng.randrange(start, end)
      samples.append( self._data[i][1] )
      idx.append( i )
      weights.append( self.importance(i) / max_w )
    return (samples, idx, weights)
  
  def _calculate_boundaries( self ):
    """ Calculate the extents of the `k` blocks. Creates `len(self._data)`
    singleton blocks if `len(self._data) <= self.k`.
    """
    N = len(self._data)
    if N <= self.k:
      self._boundaries = list(range(0, N + 1))
      self._Z = sum( math.pow(1/i, self.alpha) for i in range(1, N+1) )
    else:
      idx, Z = harmonic_spacings( N, self.k, self.alpha )
      self._boundaries = [0] + idx
      self._Z = Z
      
  def probability( self, i ):
    """ The probability of sampling the item at index `i`.
    """
    return (1 / math.pow( i+1, self.alpha )) / self._Z
    
  def importance( self, i ):
    """ The importance weight of the item at index `i`.
    """
    # Should be more stable than naive formula
    return math.pow( self._Z * math.pow(i+1, self.alpha) / len(self._data), self.beta )
    # return math.pow( 1.0 / (len(self._data) * self.probability(i)), self.beta )
    
  def max_importance( self ):
    """ The maximum importance weight among the current samples.
    """
    return self.importance( len(self._data) - 1 )
