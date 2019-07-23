import heapq
import math

def astar( model ):
  """ The A* search algorithm.
  
  Parameters:
    model: The search space
    
  Return:
    `(cost, path)`, or `None` if no solution

  Assumptions:
    * States are value types
    * Edge weights are positive

  The algorithm estimates the total cost of reaching a goal state from a
  state `s` as the sum:
    ```f(s) = g(s) + h(s)```
  where `g(s)` is the realized cost of reaching `s` and `h(s)` is a heuristic
  estimate of the cost-to-go.
  
  This version does not use a "closed list". This means it works with
  heuristics that are not consistent, but a faster implementation may be
  possible if the heuristic *is* consistent.
  """
  s0 = model.start()
  open_set = set( [s0] )              # Candidate moves
  pq = [ (model.heuristic(s0), s0) ]  # Priority queue of candidate moves
  parent = dict()                     # Least-cost paths to each node
  running_cost = dict( [(s0, 0)] )    # Realized cost of each node encountered
  
  def path( s ):
    # Extract path from `s0` to `s`
    p = []
    while True:
      p.append( s )
      if s == s0:
        break
      s = parent[s]
    return reversed( p )
  
  while len(open_set) > 0:
    while True: # Loop handles multiply-encountered states
      (cost, s) = heapq.heappop( pq )
      if model.goal( s ):
        return (cost, path( s ))
      try:
        open_set.remove( s )
      except KeyError:
        # If `s` was not in `open_set`, then it has been added more than once
        # and a lower-cost instance was already removed => Keep looking
        continue
      break
    
    for sprime in model.successors( s ):
      g = running_cost[s] + model.edge_weight( s, sprime )
      if g < running_cost.get( sprime, math.inf ):
        # Found a new optimal path to `sprime`
        open_set.add( sprime )
        parent[sprime] = s
        running_cost[sprime] = g
        f = g + model.heuristic(sprime)
        heapq.heappush( pq, (f, sprime) )
  # No solution
  assert( len(pq) == 0 )
  return None
