import itertools
import math

def proportional( samples, values, model ):
  """ Implements proportional cross-entropy.
  
  @inproceedings{goschin2013cross,
    title={The cross-entropy method optimizes for quantiles},
    author={Goschin, Sergiu and Weinstein, Ari and Littman, Michael},
    booktitle={International Conference on Machine Learning},
    pages={1193--1201},
    year={2013}
  }
  """
  weights = values[:]
  N = len(weights)
  w2 = itertools.tee( weights )
  m, M = min(w2[0]), max(w2[1])
  d = M - m
  if d == 0:
    weights = [1.0] * N
  else:
    weights = [w - m / d for w in weights]
    Z = sum(weights) # Can't be 0, or d would be 0
    weights = [w / Z for w in weights]
  return weights

def _best( samples, values, model, fraction ):
  assert( 0 < fraction )
  assert( fraction <= 1 )
  N = len(samples)
  keep = int( math.ceil(N * fraction) )
  # (value, index)
  top = sorted( zip(values, range(N)), reverse=True )
  weights = [0] * N
  for i in range(keep):
    (v, idx) = top[i]
    weights[idx] = 1.0 / keep
  return weights
  
def best( fraction ):
  """ Implements the standard "elite set" cross-entropy method.
  
  Parameters:
    `fraction` : `[0, 1]` The best `fraction` samples get uniform weight and
      the rest get 0 weight.
  """
  return lambda c, m: _best(c, m, fraction)

def cross_entropy_search( model, generations, generation_size, weight_fn = proportional ):
  """ Cross-entropy search optimizes a stochastic function by iteratively
  generating a sample set, evaluating the samples, and optimizing the
  target function to make high-scoring samples more likely.
  
  The `model` instance must implement the following operations:
    `model.propose()` : Return a sample from the target function
    `model.evaluate(samples)` : Evaluates all the samples in list `samples` and
      returns a list of values in the same order.
    `model.optimize(samples, weights)` : Improve the target function.
      `samples` contains the current sample population and `weights` is a
      probability vector of the same length.
  
  The `weight_fn` parameter controls how weights are computed. Current choices
  are `proportional` (default) and `best(fraction)`.
  
  Parameters:
    `model` : Implements domain-specific operations (see above)
    `generations` : `int` How many generations
    `generation_size` : `int` How many samples in each generation
    `weight_fn` : `[samples] -> model -> [weights]`
  """
  for g in range(generations):
    samples = [model.propose() for _ in range(generation_size)]
    values = model.evaluate( samples )
    weights = weight_fn( samples, values, model )
    model.optimize( samples, weights )
