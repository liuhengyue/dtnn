""" The HC-Search framework.
"""

import enum
import functools
import heapq
import math
import types

from nnsearch.tree_search import depth_first_search, DfsVisitor

# ----------------------------------------------------------------------------

def add_parent_value( node, model ):
  return node.parent().value() + model.heuristic( node.state() )

class AddParentValueHeuristic:
  """ Adds the `node`'s heuristic value to the value of its parent.
  """
  def __call__( self ):
    def h( self, node, model ):
      return node.parent().value() + model.heuristic( node.state() )
    return h

@enum.unique
class Heuristic(enum.Enum):
  """ Available HC-Search heuristics. Members can be called directly,
  like `Heuristic.AddParentValue( node, model )`.
  """
  AddParentValue = functools.partial( add_parent_value )
  
  def __call__( self, *args ):
    return self.value( *args )
  
  @staticmethod
  def from_json( json_string ):
    for h in Heuristic:
      if h.name == json_string:
        return h
    raise RuntimeError( "Invalid heuristic '" + json_string + "'" )

# ----------------------------------------------------------------------------
    
class ControllerBase:
  def discover_node( self, n ):
    pass
    
  def expand_node( self, n ):
    pass
    
  def terminate( self ):
    return False

class ExpandLimitController(ControllerBase):
  """ Terminates search after a fixed number of `expand` operations.
  """
  def __init__( self, limit ):
    self._expanded = 0
    self._limit = limit
    
  def expand_node( self, n ):
    self._expanded += 1
    print( "ExpandLimitController.expand_node(): " + str(self._expanded) )
    
  def terminate( self ):
    return self._expanded >= self._limit
    
@enum.unique
class Controller(enum.Enum):
  """ Available HC-Search controllers.
  """
  ExpandLimit = ExpandLimitController
  
  @staticmethod
  def from_json( json_dict ):
    for c in Controller:
      if c.name == json_dict["name"]:
        return c.value( **json_dict["parameters"] )
    raise RuntimeError( "Invalid controller '" + str(json_dict["name"]) + "'" )

# ----------------------------------------------------------------------------
    
class HcSearchAlgorithm:
  """ Encasulates the `hcsearch` algorithm with fixed parameters supplied as a
  JSON dictionary.
  """
  def __init__( self, json_dict ):
    self._parameters = json_dict
    
  def _install_heuristic( self, controller ):
    h = Heuristic.from_json( self._parameters["heuristic"] )
    controller.heuristic = types.MethodType( lambda _self, node, model: h(node, model), controller )
    
  def __call__( self, model ):
    controller = Controller.from_json( self._parameters["controller"] )
    self._install_heuristic( controller )
    return hcsearch( model, controller )

class TreeNode:
  """ Search tree node used in hcsearch.
  """
  def __init__( self, model, s, parent ):
    self._s = s
    self._parent = parent 
    self._children = []
    self._heuristic_value = -math.inf
    self._value = -math.inf
    self._expanded = False
    
  def expand( self, model, controller ):
    self._value = model.evaluate( self._s )
    for sprime in model.successors( self._s ):
      node = TreeNode( model, sprime, self )
      node._heuristic_value = controller.heuristic( node, model )
      self._children.append( node )
    self._expanded = True
      
  def expanded( self ):
    return self._expanded
    
  def parent( self ):
    return self._parent
    
  def children( self ):
    return iter(self._children)
    
  def state( self ):
    return self._s
      
  def heuristic_value( self ):
    return self._heuristic_value
    
  def value( self ):
    return self._value

def hcsearch( model, controller ):
  """ Basic implementation of HC-Search. "HC" nominally stands for
  "heuristic-cost", but we try to maximize "value" rather than minimize cost.
  
  The search tree is expanded greedily in order of heuristic estimate. When the
  search terminates, the candidates are returned in descending order of their
  value estimates.
  
  Parameters:
    model : Implements `start()`, `successors(s)`, `heuristic(s),
            `evaluate(s)`
    controller : Implements: `discover_node(n)`, `expand_node(n)`,
            `terminate() -> bool`
            
  Returns:
    `[(value, state)]` in descending order of `value`.
  """  
  root_node = TreeNode( model, model.start(), parent=None )
  leaf_pq = []
  tiebreak = 0
  def push( node ):
    nonlocal tiebreak
    heapq.heappush( leaf_pq, (-node.heuristic_value(), tiebreak, node) )
    tiebreak += 1
    controller.discover_node( node )
  def pop():
    _priority, _tiebreak, node = heapq.heappop( leaf_pq )
    return node
    
  push( root_node )
  while len(leaf_pq) > 0 and not controller.terminate():
    node = pop()
    node.expand( model, controller )
    controller.expand_node( node )
    for child in node.children():
      push( child )
    
  class ValueExtractor(DfsVisitor):
    def __init__( self ):
      self.values = []
      self._tiebreak = 0
  
    def discover_vertex( self, v ):
      if v.expanded():
        self.values.append( (v.value(), self._tiebreak, v.state()) )
        self._tiebreak += 1
  
  vex = ValueExtractor()
  depth_first_search( root_node, vex, lambda v: v.children() )
  vex.values.sort( reverse=True )
  return [(value, state) for (value, _, state) in vex.values]
