""" Multi-armed bandit algorithms.
"""

import abc
import heapq
import math

from nnsearch.statistics import MeanAccumulator

class BanditRule(metaclass=abc.ABCMeta):
  """ Base class for arm selection rules. Implementations should be stateless.
  """
  
  @abc.abstractmethod
  def next_arm( self, arms, n, context ):
    """ Select the next "arm" to pull.
    
    Parameters:
      `arms` : iterable of arbitrary "arm" objects
      `n` : Which sample is currently being drawn (should be 1 for first pull)
      `context` : Provides any algorithm-specific functions or data
    """
    return None

class MaxBanditRule(BanditRule):
  """ Selects the arm that maximizes a heuristic.
  """
  
  @abc.abstractmethod
  def heuristic( self, a, n, context ):
    """ The heuristic value of an arm.
    
    Parameters:
      `a` : The arm object
      `n` : Current pull number (should be 1 for first pull)
      `context` : Provides algorithm-specific functions or data
    """
    raise NotImplementedError
  
  def next_arm( self, arms, n, context ):
    astar = []
    vstar = -math.inf
    for a in arms:
      h = self.heuristic( a, n, context )
      # if h == math.inf:
        # return a
      if h > vstar:
        vstar = h
        astar = [a]
      elif h == vstar:
        astar.append( a )
    return astar

class Ucb1(MaxBanditRule):
  """ The UCB-1 rule. Standard algorithm for cumulative regret.
  
  @article{auer2002finite,
    title={Finite-time analysis of the multiarmed bandit problem},
    author={Auer, Peter and Cesa-Bianchi, Nicolo and Fischer, Paul},
    journal={Machine learning},
    volume={47},
    number={2-3},
    pages={235--256},
    year={2002}
  }
  """
  
  def __init__( self, c = 1.0 ):
    """
    Parameters:
      `narms` : Number of arms
      `c` : The "exploration constant" used in some extensions of UCB (such as
        the UCT algorithm)
    """
    super().__init__()
    self._c = c
    
  def heuristic( self, a, n, context ):
    na = context.n(a)
    if na == 0:
      return math.inf
    else:
      assert( n > 0 )
      return context.value(a) + self._c * math.sqrt(2 * math.log(n) / na)
      
class UcbSqrt(MaxBanditRule):
  """ Ucb-Sqrt rule. Better for simple regret than Ucb1.
  
  @inproceedings{tolpin2012mcts,
    title={MCTS Based on Simple Regret.},
    author={Tolpin, David and Shimony, Solomon Eyal},
    booktitle={AAAI},
    year={2012}
  }
  """
  def __init__( self, c = 1 ):
    """
    Parameters:
      `narms` : Number of arms
      `c` : The "exploration constant". Note that we put the exploration
        constant `c` outside the outer square-root so that its relationship to
        the reward magnitude is easier to understand.
    """
    super().__init__()
    self._c = c
  
  def heuristic( self, a, n, context ):
    na = context.n(a)
    if na == 0:
      return math.inf
    else:
      assert( n > 0 )
      return context.value(a) + self._c * math.sqrt( math.sqrt(n) / na )

# ----------------------------------------------------------------------------

class Bandit:
  """ A bandit over `range(narms)`.
  """
  
  def __init__( self, bandit_rule, narms ):
    self._rule = bandit_rule
    self._ravg = [0] * narms
    self._ns = [0] * narms
    self._n = 0
  
  def narms( self ):
    return len(self._ns)
    
  def candidates( self ):
    return list(range(self.narms()))
    
  def next_arm( self ):
    return self._rule.next_arm( range(self.narms()), self._n, self )
    
  def update( self, i, r ):
    """ Update arm statistics after a pull.
    
    Parameters:
      `i` : Arm index
      `r` : Reward received
    """
    self._n       += 1
    self._ns[i]   += 1
    self._ravg[i] += (r - self._ravg[i]) / self._ns[i]
    
  def n( self, i ):
    return self._ns[i]
    
  # def n( self ):
    # return self._n
  
  def value( self, i ):
    return self._ravg[i]

# ----------------------------------------------------------------------------

class Hkr1Bandit:
  def __init__( self, narms, budget ):
    self._T = budget
    self._t = 0
    self._narms = narms
    self._S = list(range(narms))
    self._s = 0
    self._F = [0] * narms
    self._f1 = [0] * narms
    self._f0 = [0] * narms
    
  def narms( self ):
    return self._narms
    
  def timeout( self ):
    return self._t >= self._T
  
  def singleton( self ):
    if len(self._S) == 1:
      return self._S[0]
    else:
      return None
  
  def candidates( self ):
    return self._S[:]
  
  def next_arm( self ):
    if len(self._S) == 1:
      return self._S[0]
    
    if self._s == len(self._S):
      # Phase finished
      self._t += 1
      self._s = 0
      
      # Compute value bounds
      p, q = [], []
      for i in self._S:
        T, t, f0, f1, F = self._T, self._t, self._f0[i], self._f1[i], self._F[i]
        # Upper bound: extrapolate last improvement
        p.append( F + sum( min(1, (f1 + (f1 - f0))*(s - t)) for s in range(t+1, T+1) ) )
        # Lower bound: assume no more improvement
        q.append( F + f1*(T - t) )
      # Drop arms that are guaranteed not optimal
      to_drop = set()
      for i in range(len(self._S)):
        def drop( i ):
          for j in range(len(self._S)):
            if i != j and q[j] > p[i]:
              return True
          return False
        if drop(i):
          to_drop.add( i )
      self._S = [a for a in self._S if a not in to_drop]
    
    # Next arm
    a = self._S[self._s]
    self._s += 1
    return [a]
      
  def update( self, i, r ):
    assert( r >= 0 )
    assert( r <= 1 )
    self._f0[i] = self._f1[i]
    self._f1[i] = r
    self._F[i] += r
    
# ----------------------------------------------------------------------------

class SequentialHalvingBandit:
  """ Allocates a fixed budget uniformly to `ceil(log2(narms))` "rounds", each 
  of which eliminates half the candidates.
  
  @inproceedings{karnin2013almost,
    title={Almost optimal exploration in multi-armed bandits},
    author={Karnin, Zohar and Koren, Tomer and Somekh, Oren},
    booktitle={International Conference on Machine Learning},
    pages={1238--1246},
    year={2013}
  }
  """
  def __init__( self, narms, budget, cumulative_mean=False ):
    self._narms = narms
    self._ns = [0] * narms
    self._B = budget
    self._S = list(range(self._narms))
    self._samples_this_round = 0
    self._next_arm = 0
    # self._nrounds = int( math.ceil(math.log(narms, 2)) ) - 1
    self._Br = self._round_budget()
    self._losses = [MeanAccumulator() for _ in range(narms)]
    self._cumulative_mean = cumulative_mean
    
  def _round_budget( self ):
    return self._B // (len(self._S) * int(math.ceil(math.log(self._narms, 2))))
  
  def narms( self ):
    return self._narms
    
  def n( self, i ):
    return self._ns[i]
    
  # def timeout( self ):
    # return self._t >= self._T
  
  def singleton( self ):
    if len(self._S) == 1:
      return self._S[0]
    else:
      return None
  
  def candidates( self ):
    return self._S[:]
  
  def next_arm( self ):
    if len(self._S) == 1:
      return [self._S[0]]
    if self._samples_this_round == len(self._S) * self._Br:
      # Finished round. Retain top half of arms
      Sr = [(-self._losses[i].mean(), i) for i in self._S]
      heapq.heapify( Sr )
      Nr = (len(self._S) + 1) // 2
      self._S = [heapq.heappop(Sr)[1] for _ in range(Nr)]
      if not self._cumulative_mean:
        for i in self._S:
          self._losses[i] = MeanAccumulator()
      assert( self._next_arm == 0 )
      self._Br = self._round_budget()
      self._samples_this_round = 0
    a = self._S[self._next_arm]
    self._samples_this_round += 1
    self._next_arm = (self._next_arm + 1) % len(self._S)
    return [a]
    
  def update( self, i, r ):
    self._losses[i]( r )
    self._ns[i] += 1

# ----------------------------------------------------------------------------

# class MapBandit:
  # def __init__( self, bandit_rule, objects ):
    # self._rule = bandit_rule( len(objects) )
    # self._objects = objects
  
  # def narms( self ):
    # return self._rule.narms()
    
  # def next_arm( self ):
    # i = self._rule.next_arm()
    # return (i, self._objects[i])
    
  # def update( self, arm, r ):
    # i, _o = arm
    # self._rule.update( i, r )
    
  # def n( self, arm ):
    # i, _o = arm
    # return self._rule.n( i )
    
  # def n( self ):
    # return self._rule.n()
    
if __name__ == "__main__":
  class LogarithmicAccumulator:
    def __init__( self, base, Z ):
      self._base = base
      self._x = 1
      self._Z = Z
      
    def __call__( self ):
      y = math.log( self._x, self._base )
      self._x += 1
      return y / self._Z
  
  T = 256
  arms = [LogarithmicAccumulator( b, 10 ) for b in range(2, 11)]
  # bandit = Hkr1Bandit( len(arms), T )
  bandit = SequentialHalvingBandit( len(arms), T )
  
  step = 0
  # while not bandit.timeout() and bandit.singleton() is None:
  while bandit.singleton() is None:
    print( "candidates: {}".format( str(bandit.candidates()) ) )
    i = bandit.next_arm()[0]
    r = arms[i]()
    print( "{}: {} -> {}".format( step, i, r ) )
    bandit.update( i, r )
    step += 1
  print( "singleton: {}".format( bandit.singleton() ) )
  
  from nnsearch.statistics import MovingMeanAccumulator
  m = MovingMeanAccumulator( 4 )
  xs = list(range(0, 11))
  for x in xs:
    m(x)
    print( m.mean() )
  