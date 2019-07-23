def isample( rng, iterable, k ):
  """ Sample `k` items without replacement from `iterable` of unknown finite
  length, using random stream `rng`.
  """
  results = []
  itr = iter(iterable)
  try:
    for i in range(k):
      results.append( next(itr) )
  except StopIteration:
    raise ValueError( "sample size larger than population" )
  rng.shuffle( results )
  for i, v in enumerate(itr, k):
    r = rng.randint(0, i)
    if r < k:
      results[r] = v
  return results
  
def ichoice( rng, iterable ):
  """ Choose an item uniformly at random from `iterable` of unknown finite
  length, using random stream `rng`.
  
  Equivalent to `isample( rng, iterable, 1 )`.
  """
  return isample( rng, iterable, 1 )
  
def multinomial( rng, ps ):
  """ Sample an index from a multinomial distribution defined by a probability
  vector.
  
  Parameters:
    `rng` : random.Random instance
    `ps` : `[float]` probability of each index (`sum(ps) == 1`)
  """
  u = rng.random()
  s = 0.0
  for (i, p) in enumerate(ps):
    s += p
    if u <= s:
      return i
  return len(ps) - 1
