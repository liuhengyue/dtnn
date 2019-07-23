import logging

import nnsearch.bandit
from nnsearch.statistics import MovingMeanAccumulator

class BanditSearch:
  def __init__( self, models, bandit, pool, gpu_logger=None ):
    """
    Parameters:
      `models` : A list of `LayerSpec`s
      `bandit` : Model of `nnsearch.bandit.Bandit` concept
      `pool` : Worker pool for evaluations
      `energy_factor` : If using energy logging, reward is calculated as:
          `(1.0 - validation_error) / (energy_factor * energy)`
        where `energy` is measured in Joules.
    """
    self._models = models
    self._bandit = bandit
    self._pool = pool
    self._gpu_logger = gpu_logger
    self._values = [MovingMeanAccumulator(3) for _ in models]
    log = logging.getLogger( __name__ )
    log.info( "Models:" )
    for (i, m) in enumerate(self._models):
      log.info( "%s: %s", i, str(m) )
    
  def search( self, rng, nsteps, value_fn, nepoch=1, warmup=0, test=False ):
    log = logging.getLogger( __name__ )
    
    for step in range(warmup):
      for (i, model) in enumerate(self._models):
        log.info( "warmup=%s;pull=(%s,%s)", step, i, str(model) )
        f = self._pool.train_and_evaluate(
          model, nepoch=1, test=test, gpu_logger=self._gpu_logger )
        performance = f.result()
        log.info( "warmup=%s;performance=%s", step, str(performance) )
        reward = value_fn( performance )
        self._values[i]( reward )
        log.info( "warmup=%s;reward=%s;model=%s", step, reward, str(model) )
    
    for step in range(nsteps):
      arm = rng.choice( self._bandit.next_arm() )
      model = self._models[arm]
      log.info( "step=%s;pull=(%s,%s)", step, arm, str(model) )
      f = self._pool.train_and_evaluate(
        model, nepoch=nepoch, test=test, gpu_logger=self._gpu_logger )
      performance = f.result()
      log.info( "step=%s;performance=%s", step, str(performance) )
      reward = value_fn( performance )
      self._bandit.update( arm, reward )
      self._values[arm]( reward )
      log.info( "step=%s;reward=%s;model=%s", step, reward, str(model) )
      for i in sorted( self._bandit.candidates() ):
        log.info( "step=%s;arm=%s;n=%s;value=%s", step, i, self._bandit.n(i), self._values[i].mean() )
