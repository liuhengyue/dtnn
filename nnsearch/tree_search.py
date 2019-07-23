def depth_first_search( vertex, visitor, neighbor_fn ):
  """ Depth-first search.
  
  Parameters:
    `vertex` : Start vertex
    `visitor` : Must implement two functions:
      `visitor.discover_vertex( u )` : Called the first time `u` is encountered
      `visitor.finish_vertex( u )`   : Called after `finish_vertex()` has been
                                       called on all successors of `u`.
    `neighbor_fn` : Function yielding neighbors of a vertex.
  """
  visited = set()
  
  def impl( u ):
    n = len(visited)
    visited.add( u )
    if len(visited) > n: # Novel vertex
      visitor.discover_vertex( u )
      for v in neighbor_fn( u ):
        impl( v )
      visitor.finish_vertex( u )
      
  impl( vertex )

class DfsVisitor:
  def discover_vertex( self, v ):
    """ Called when encountering a novel vertex.
    """
    pass
  
  def finish_vertex( self, v ):
    """ Called after `finish_vertex()` has been called on all descendants of
    `v` in the DFS tree.
    """
    pass
