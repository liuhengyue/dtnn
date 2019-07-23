""" Implements a generic Monte Carlo tree search framework, which can be
instantiated to create algorithms such as UCT.
"""

from collections import OrderedDict
import enum
import logging
import math
import random

import nnsearch.bandit
import nnsearch.sample as sample
from nnsearch.tree_search import depth_first_search

class StateNode:
  """ State node in MCTS tree.
  """
  
  def __init__( self, s, controller ):
    """
    Parameters:
      `s` : The state associated with this node
      `controller` : Provides algorithm-specific behaviors
    """
    self._s = s
    self._n = 0
    self._successors = OrderedDict()
    self._heuristic_value = -math.inf
    self._expanded = False
    self._bandit_rule = controller.bandit_rule()
  
  def visit( self ):
    self._n += 1
    
  def maxq( self ):
    assert( not self.leaf() )
    q = -math.inf
    for an in self.successors():
      q = max( an.value(), q )
    return q
    
  def n( self ):
    return self._n
    
  def choose_action( self, controller ):
    an = self._bandit_rule.next_arm( self.successors(), self.n(), ActionNode )
    if len(an) == 1 or controller.rng() is None:
      return an[0]
    else:
      return controller.rng().choice( an )
    # return an
    
  def leaf( self ):
    return len(self._successors) == 0
    
  def expand( self, model, controller, path ):
    assert( not self.expanded() )
    for a in model.actions( self._s ):
      an = ActionNode( a, controller.qinit( path, a ) )
      self._successors[a] = an
    # FIXME: Consider removing heuristic_value field
    self._heuristic_value = controller.evaluate( path, self )
    self._expanded = True
    return self._heuristic_value
      
  def expanded( self ):
    return self._expanded
    
  def successors( self ):
    return self._successors.values()
    
  def successor( self, a ):
    return self._successors[a]
    
  def state( self ):
    return self._s
      
  def heuristic_value( self ):
    return self._heuristic_value
    
  def __str__( self ):
    return "SN {" + ", ".join(
      ["s: " + str(self._s), "n: " + str(self._n)] ) + "}"

class ActionNode:
  """ Action node in MCTS tree.
  """
  
  def __init__( self, a, qinit = 0 ):
    """
    Parameters:
      `a` : The action associated with this node
    """
    self._a = a
    self._n = 0
    self._q = qinit
    self._successors = OrderedDict()
  
  def action( self ):
    return self._a
    
  def visit( self ):
    """ Increment visit count. Call before `qsample()`.
    """
    self._n += 1
    
  def n( self ):
    """ Visit count.
    """
    return self._n
  
  def value( self ):
    """ The current value (i.e. Q-value) estimate for this action.
    """
    return self._q
    
  def qsample( self, q ):
    """ Provide a new sample of the Q-value of this action in the context of
    its parent state node. Call after `visit()`.
    """
    self._q += (q - self._q) / self.n()
    
  def successors( self ):
    return self._successors.values()
    
  def successor( self, s ):
    return self._successors[s]
    
  def require_successor( self, s, controller ):
    """ Get successor, or create it if it doesn't exist.
    """
    try:
      snprime = self.successor( s )
    except KeyError:
      snprime = StateNode( s, controller )
      self._successors[s] = snprime
    return snprime
    
  def __str__( self ):
    return "AN {" + ", ".join(
      ["a: " + str(self._a), "q: " + str(self._q), "n: " + str(self._n)] ) + "}"

class TreePrinterDfs:
  def __init__( self, indent=2, print_fn = print ):
    self._indent = " " * indent
    self._depth = 0
    self._print_fn = print_fn
  
  def discover_vertex( self, v ):
    self._print_fn( self._indent * self._depth + str(v) )
    self._depth += 1
    
  def finish_vertex( self, v ):
    self._depth -= 1
    
def print_tree( sn0, print_fn = print ):
  depth_first_search( sn0, TreePrinterDfs( print_fn=print_fn ), lambda v: v.successors() )

# ----------------------------------------------------------------------------

class UniformRandomPolicy:
  def __call__( self, rng, model, s ):
    a = sample.ichoice( rng, model.actions( s ) )
    return a

class RolloutEvaluator:
  def __init__( self, policy, depth = math.inf ):
    self._policy = policy
    self._depth = depth

  def __call__( self, rng, model, s ):
    gamma = self._model.discount()
    q = 0
    d = 0
    while d < self._depth and not model.terminal( s ):
      a = self._policy( rng, model, s )
      (s, r) = model.sample_transition( rng, s, a )
      q += r * gamma
      gamma *= model.discount()
      d += 1
    return q
    
class ConstantEvaluator:
  def __init__( self, v ):
    self._v = v
    
  def __call__( self, rng, model, controller, path, sn ):
    return self._v
    
class ParentEvaluator:
  def __init__( self, v ):
    self._v = v
    
  def __call__( self, rng, model, controller, path, sn ):
    if len(path) > 1:
      return path[-3].maxq()
    else:
      return self._v
    
# ----------------------------------------------------------------------------

class UctController:
  """ Implements the standard UCT algorithm: Monte Carlo updates, Ucb1 bandit
  rule, random rollouts for leaf evaluation.
  """
  
  def __init__( self, rng, model, trajectory_limit, evaluator, c = 1, qinit = 0 ):
    """
    Parameters:
      
    """
    self._rng = rng
    self._model = model
    self._trajectory_limit = trajectory_limit
    self._ntrajectories = 0
    self._c = c
    self._qinit = qinit
    self._evaluator = evaluator
  
  def rng( self ):
    return self._rng
  
  def begin_trajectory( self, sn0 ):
    if self._ntrajectories >= self._trajectory_limit:
      return False
    else:
      self._ntrajectories += 1
      return True
      
  def finish_trajectory( self, sn0 ):
    pass
  
  def update( self, sn, an, snprime, q ):
    an.qsample( q )
    
  def bandit_rule( self ):
    return nnsearch.bandit.Ucb1( self._c )
    
  def evaluate( self, path, sn ):
    return self._evaluator( self._rng, self._model, self, path, sn )
    
  def qinit( self, path, a ):
    return self._qinit
    
class MaxUctController(UctController):
  def __init__( self, rng, model, trajectory_limit, evaluator, c = 1, qinit = 0 ):
    super().__init__( rng, model, trajectory_limit, evaluator, c, qinit )
    
  def update( self, sn, an, snprime, q ):
    if snprime.leaf():
      an.qsample( q )
    else:
      qmax = -math.inf
      for anprime in snprime.successors():
        qmax = max( qmax, anprime.value() )
      an.qsample( self._model.discount() * qmax )

# ----------------------------------------------------------------------------

class MctsAlgorithm:
  """ Encasulates the `mcts` algorithm with fixed parameters supplied as a
  JSON dictionary.
  """
  def __init__( self, json_dict ):
    self._parameters = json_dict

  def __call__( self, model ):
    controller = Controller.from_json( self._parameters["controller"], model )
    return mcts( model, controller )
    
# ----------------------------------------------------------------------------

def trajectory( model, controller, root, choose_action ):
  """ Performs one MCTS sampling trajectory.
  """
  logger = logging.getLogger( __name__ )
  sn = root
  path = [sn]
  rewards = []
  
  # Run "tree policy" until un-expanded state node
  while not sn.leaf():
    sn.visit()
    an = choose_action( sn, controller )
    an.visit()
    path.append( an )
    (sprime, r) = model.sample_transition( controller.rng(), sn.state(), an.action() )
    sn = an.require_successor( sprime, controller )
    path.append( sn )
    rewards.append( r )
  logger.info( "tree path %s", [str(n) for n in path] )
  logger.info( "rewards %s", rewards )
  
  # sn is a leaf
  assert( len(path) % 2 == 1 )
  snprime = path[-1]
  if not model.terminal( snprime.state() ):
    q = snprime.expand( model, controller, path )
    logger.info( "expand %s", snprime )
    # q = snprime.heuristic_value()
  else:
    q = 0
  
  # Back-propagate value
  snprime = path.pop()
  while len(path) > 0:
    an = path.pop()
    sn = path.pop()
    r = rewards.pop()
    q = r + model.discount() * q
    controller.update( sn, an, snprime, q )
    snprime = sn
  return q
    
def mcts( model, controller ):
  """ Generic Monte Carlo tree search.
  """
  root = StateNode( model.start(), controller )
  while controller.begin_trajectory( root ):
    trajectory( model, controller, root,
                choose_action = lambda sn, ctrl: sn.choose_action( ctrl ) )
    controller.finish_trajectory( root )
  return root

def mcts2( model, controller ):
  """ Generic Monte Carlo tree search.
  """
  logger = logging.getLogger( __name__ )
  root = StateNode( model.start(), controller )
  while controller.begin_trajectory( root ):
    sn = root
    path = [sn]
    rewards = []
    
    # Run "tree policy" until un-expanded state node
    while not sn.leaf():
      sn.visit()
      an = sn.choose_action()
      an.visit()
      path.append( an )
      (sprime, r) = model.sample_transition( controller.rng(), sn.state(), an.action() )
      sn = an.require_successor( sprime, controller )
      path.append( sn )
      rewards.append( r )
    logger.info( "tree path %s", [str(n) for n in path] )
    logger.info( "rewards %s", rewards )
    
    # sn is a leaf
    assert( len(path) % 2 == 1 )
    snprime = path[-1]
    if not model.terminal( snprime.state() ):
      q = snprime.expand( model, controller, path )
      logger.info( "expand %s", snprime )
      # q = snprime.heuristic_value()
    else:
      q = 0
    
    # Back-propagate value
    snprime = path.pop()
    while len(path) > 0:
      an = path.pop()
      sn = path.pop()
      r = rewards.pop()
      q = r + model.discount() * q
      controller.update( sn, an, snprime, q )
      snprime = sn
    controller.finish_trajectory( root )
  # Return search tree
  return root
