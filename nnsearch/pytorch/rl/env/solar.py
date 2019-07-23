from   collections import namedtuple
import itertools
import logging
import math

import numpy as np

import torch
from   torch.autograd import Variable
import torch.nn.functional as fn
from   torch.utils.data import DataLoader

import gym
import gym.spaces

from   nnsearch.pytorch import rl
from   nnsearch.pytorch import torchx
from   nnsearch.pytorch.gated.module import GatedModule
from   nnsearch.pytorch.rl import gymx
from   nnsearch.statistics import MovingMeanAccumulator

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

class PiecewiseLinearSunlightModel:
  """ Nighttime is assumed to last 12 hours, from 1800-0600. Sun intensity
  ramps up linearly from 0600-0800, stays at max intensity from 0800-1600, and
  ramps down linearly from 1600-1800.
  """
  def __init__( self, rng, steps_per_hour=60 ):
    self.rng = rng
    self.steps_per_hour = steps_per_hour
    self.sun_ramp = self.steps_per_hour * 2
    self.half_night = self.steps_per_hour * 6
    self.t = 0
    self.T = self.steps_per_hour * 24
    self.reset()
    
  @property
  def intensity( self ):
    full_daylight = 24 - 2*self.half_night - 2*self.sun_ramp
    if self.t < self.half_night: # Darkness
      return 0.0
    tday = self.t - self.half_night
    if tday < self.sun_ramp: # Increasing intensity
      return tday / self.sun_ramp
    tday -= self.sun_ramp
    if tday < full_daylight: # Max intensity
      return 1.0
    tday -= full_daylight
    if tday < self.sun_ramp: # Decreasing intensity
      return 1.0 - (tday / self.sun_ramp)
    tday -= self.sun_ramp
    assert tday <= half_night
    return 0.0 # Darkness
    
  def reset( self, randomize=True ):
    self.t = self.rng.randrange(self.T) if randomize else 0
    
  def step( self ):
    self.t += 1
    if self.t >= self.T:
      self.t = 0
    log.debug( "SunlightModel.hour: %s", self.t / self.steps_per_hour )

class SinusoidSunlightModel:
  """ Sun intensity follows a truncated sinusoid with max intensity at noon
  and zero intensity from 1800-0600.
  """
  def __init__( self, rng, steps_per_hour=60 ):
    self.rng = rng
    self.steps_per_hour = steps_per_hour
    self.t = 0
    self.T = self.steps_per_hour * 24
    self.reset()
    
  @property
  def intensity( self ):
    return max( 0, math.cos(math.pi * (1 + 2*self.t/self.T)) )
    
  def reset( self, randomize=True ):
    self.t = self.rng.randrange(self.T) if randomize else 0
    
  def step( self ):
    self.t += 1
    if self.t >= self.T:
      self.t = 0
    log.debug( "SunlightModel.hour: %s", self.t / self.steps_per_hour )
      
# ----------------------------------------------------------------------------

class UniformCloudModel:
  def __init__( self, rng, minval=0, maxval=1 ):
    self.rng = rng
    self._minval = minval
    self._maxval = maxval
    
  @property
  def minval( self ):
    return self._minval
  
  @property
  def maxval( self ):
    return self._maxval
  
  def reset( self ):
    self.cloudiness = self.rng.uniform( self._minval, self._maxval )
    
  def step( self ):
    self.cloudiness = self.rng.uniform( self._minval, self._maxval )

class RandomWalkCloudModel:
  """ Cloudiness follows a bounded random walk on [0, 1] with Gaussian
  increments.
  """
  def __init__( self, rng, minval=0, maxval=1, drift=0, sigma=0.02 ):
    self.rng = rng
    self._minval = minval
    self._maxval = maxval
    self.drift = drift
    self.sigma = sigma
    self.cloudiness = 0
    self.reset()
    
  @property
  def minval( self ):
    return self._minval
  
  @property
  def maxval( self ):
    return self._maxval
  
  def reset( self ):
    self.cloudiness = self.rng.uniform( self._minval, self._maxval )
    
  def step( self ):
    eps = self.rng.gauss(0, self.sigma)
    self.cloudiness = max(self._minval, 
                          min(self._maxval, self.cloudiness + self.drift + eps))
    
class HierarchicalCloudModel:
  """ Cloudiness follows a random walk with drift, where the drift is
  determined by a hidden variable that transitions between two states
  (conceptually "high" and "low" barometric pressure) following a geometric
  distribution. In the low pressure state, drift pulls cloudiness toward a
  nominal value of 0.8, and in the high pressure state towards 0.2.
  """
  def __init__( self, rng, drift_factor=0.01, sigma=0.02,
                storm_shortness_factor=1, trans_prob=(1 / 5760) ):
    """
    Parameters:
    -----------
      `drift_factor`: The drift is `drift_factor*(target - current)`
      `sigma`: Variance of the random walk
      `storm_shortness_factor`: How much shorter "stormy" periods are relative
        to "sunny" periods
      `trans_prob`: Base probability of changing from "sunny" to "stormy"
        weather every step. Default value assumes 1-minute time steps and gives
        expected "sunny" period length of 4 days.
    """
    self.rng = rng
    self.drift_factor = drift_factor
    self.sigma = sigma
    self.storm_shortness_factor = storm_shortness_factor
    self.trans_prob = trans_prob
    self.hidden = 0
    self._drift = 0
    self.cloudiness = 0
    self.reset()
  
  def reset( self ):
    self.cloudiness = self.rng.uniform( 0, 1 )
    self.hidden = self.rng.randrange(2)
    
  def step( self ):
    if self.hidden == 0:
      target = 0.2
      if self.rng.random() < self.trans_prob:
        self.hidden = 1
    else:
      target = 0.8
      if self.rng.random() < self.storm_shortness_factor*self.trans_prob:
        self.hidden = 0
    d = target - self.cloudiness
    drift = self.drift_factor * d
    eps = self.rng.gauss(0, self.sigma)
    self.cloudiness = max(0, min(1, self.cloudiness + drift + eps))

# ----------------------------------------------------------------------------

class ExposureModel:
  """ A simple model of radiant exposure from sunlight. Sun intensity is
  modulated by cloud cover, according to:
    exposure = (0.1 + 0.9*(1 - cloudiness)) * sun_intensity.
  
  Using the SinusoidSunlightModel and either the RandomWalkCloudModel or the
  HierarchicalCloudModel with default parameters, the average total sun
  exposure per day is equivalent to about 4.1-4.2 hours of max intensity, which
  is in line with averages for the northeast continental US. See e.g.: 
  https://www.wholesalesolar.com/solar-information/sun-hours-us-map
  """
  def __init__( self, sunlight_model, cloud_model ):
    self.sunlight_model = sunlight_model
    self.cloud_model = cloud_model
    
  @property
  def t( self ):
    return self.sunlight_model.t
    
  @property
  def T( self ):
    return self.sunlight_model.T
    
  @property
  def exposure( self ):
    return ( (0.1 + 0.9*(1.0 - self.cloud_model.cloudiness))
             * self.sunlight_model.intensity )
  
  def reset( self, randomize=True ):
    self.sunlight_model.reset( randomize=randomize )
    self.cloud_model.reset()
    
  def step( self ):
    self.sunlight_model.step()
    self.cloud_model.step()
    log.debug( "ExposureModel.exposure: %s (sun: %s; clouds: %s)",
      self.exposure, self.sunlight_model.intensity, self.cloud_model.cloudiness)
    
# ----------------------------------------------------------------------------

class SolarBatteryModel:
  def __init__( self, rng, exposure_model, capacity_J, max_generation_mW,
                seconds_per_step=60 ):
    self.rng = rng
    self.exposure_model = exposure_model
    self.seconds_per_step = seconds_per_step
    self.capacity_J = capacity_J
    self.max_generation_mW = max_generation_mW
    self.energy = None
    
  def reset( self, randomize=True ):
    self.exposure_model.reset( randomize=randomize )
    self.energy = self.capacity_J
    if randomize:
      self.energy *= self.rng.uniform(0, 1)
    
  def step( self, drain ):
    exposure = self.exposure_model.exposure
    charge = exposure*self.seconds_per_step*self.max_generation_mW*1e-3
    e = self.energy - drain + charge
    e = max(0, min(self.capacity_J, e))
    log.debug( "SolarBatteryModel.energy: %s - %s + %s -> %s",
               self.energy, drain, charge, e )
    self.energy = e
    self.exposure_model.step()

  @property
  def charge( self ):
    return self.energy / self.capacity_J
    
class TegraK1:
  def __init__( self ):
    # https://wccftech.com/nvidia-tegra-k1-performance-power-consumption-revealed-xiaomi-mipad-ship-32bit-64bit-denver-powered-chips/
    self.J_per_GFLOP = 1 / 26.0
    
  def flop_energy( self, nflop ):
    return self.J_per_GFLOP * (nflop / 1e9)

# ----------------------------------------------------------------------------
    
LabelReward = namedtuple( "LabelReward", ["incorrect", "correct"] )
    
class UniformRewardModel:
  def __init__( self, Rcorrect=0, Rincorrect=-1 ):
    self._Rcorrect = Rcorrect
    self._Rincorrect = Rincorrect
    
  def reward( self, y, yhat ):
    assert y.nelement() == 1
    if yhat is None:
      return self._Rincorrect
    assert yhat.nelement() == 1
    correct = y == yhat
    return self._Rcorrect if correct else self._Rincorrect
    
  def Rmin( self ):
    return min(self._Rcorrect, self._Rincorrect)
    
  def Rmax( self ):
    return max(self._Rcorrect, self._Rincorrect)
    
class WeightedRewardModel:
  def __init__( self, label_rewards ):
    assert all( isinstance(t, LabelReward) for t in label_rewards )
    self._label_rewards = label_rewards
    self._Rmin = min(self._label_rewards)
    self._Rmax = max(self._label_rewards)
    
  def reward( self, y, yhat ):
    assert y.nelement() == 1
    i = y.item()
    rs = self._label_rewards[i]
    if yhat is None:
      return rs.incorrect
    assert yhat.nelement() == 1
    correct = y == yhat
    return rs.correct if correct else rs.incorrect
    
  def Rmin( self ):
    return self._Rmin
    
  def Rmax( self ):
    return self._Rmax
    
class ConfusionRewardModel:
  def __init__( self, confusion_matrix, none_prediction=None ):
    """
    Parameters:
    -----------
      `confusion_matrix` : `confusion_matrix[i][j]` is the reward for predicting
        label `j` when the true label is `i`.
      `none_prediction` : If `None` is a valid prediction, this should be a
        vector where `none_prediction[i]` is the reward for predicting `None`
        when the true label is `i`.
    """
    self._confusion_matrix = confusion_matrix
    self._none_prediction = none_prediction
    self._Rmin = min( min(self._none_prediction),
                      min(cr for row in self._confusion_matrix for cr in row) )
    self._Rmax = max( max(self._none_prediction),
                      max(cr for row in self._confusion_matrix for cr in row) )
    
  def reward( self, y, yhat ):
    assert y.nelement() == 1
    i = y.item()
    if yhat is None:
      return self._none_prediction[i]
    assert yhat.nelement() == 1
    ihat = yhat.item()
    return self._confusion_matrix[i][ihat]
    
  def Rmin( self ):
    return self._Rmin
    
  def Rmax( self ):
    return self._Rmax

# ----------------------------------------------------------------------------
    
class ImageClassificationProblem:
  def __init__( self, battery_model, hardware_model, network, dataset,
                data_directory, reward_model, to_device, train, controller_macc=None ):
    self.battery_model = battery_model
    self.hardware_model = hardware_model
    self.reward_model = reward_model
    self.network = network
    self.to_device = to_device
    self._train = train
    # self.t_subdivision = t_subdivision
    self.dataset = dataset
    data = dataset.load( root=data_directory, train=train )
    self.loader = DataLoader(
      data, batch_size=1, shuffle=True, pin_memory=True )
    self._data_itr = None
    self._current = None
    # flops() is currently only implemented for GatedDenseNet
    total_macc, gated_flops = self.network.flops( dataset.in_shape )
    gated_total = sum( c.macc for m in gated_flops for c in m )
    self.ungated_data_macc = total_macc - gated_total
    self.controller_macc = controller_macc
    # list of 1xC tensors containing macc count per gated component
    self.gated_macc = [
      self.to_device( torch.tensor( [c.macc for c in m] ).unsqueeze(0) )
      for m in gated_flops]
    
  def _advance( self ):
    try:
      x, y = next(self._data_itr)
      x = self.to_device( x )
      y = self.to_device( y )
      self._current = (x, y)
      return False
    except StopIteration:
      return True
  
  def _tensor( self, x ):
    x = [x] if np.isscalar( x ) else x
    t = torch.FloatTensor( x )
    t = t.unsqueeze(0)
    return self.to_device(t)
  
  def energy_consumption( self, gs ):
    # FIXME: Need to verify that this is giving the right answer
  
    # Elements of 'gs' are tuples; 2nd component contains data needed for
    # training the gate functions
    macc = self.controller_macc
    if gs is not None:
      gated_macc = [torch.sum(g[0].data * n, dim=1) 
                    for (g, n) in zip(gs, self.gated_macc)]
      gated_macc = sum( gated_macc )
      gated_macc += self.ungated_data_macc
      gated_macc = torch.sum( gated_macc ).item()
      macc += gated_macc
    nflop = 2*macc
    log.debug( "problem.energy_consumption.flops: %s", nflop )
    return self.hardware_model.flop_energy( nflop )
    
  def reset( self ):
    self.battery_model.reset( randomize=self._train )
    self._data_itr = iter(self.loader)
    terminal = self._advance()
    assert not terminal
    x, _y = self._current
    # Outputs from the data load are already tensors
    return x
      
  def step( self, u ):
    log.debug( "problem.step.u: %s %s", u, type(u) )
    x, y = self._current
    log.debug( "problem.step.x: %s", x )
    log.debug( "problem.step.x.size(): %s", x.size() )
    log.debug( "problem.step.y: %s", y.item() )
    if u is None: # Network turned off
      log.debug( "problem.step.yhat: None" )
      yhat = None
      r = self.reward_model.reward( y, yhat )
      gs = None
    else:
      u = self.to_device( torch.tensor( [float(u)] ) )
      # u = self._tensor( u )
      yhat, gs = self.network( torchx.Variable(x), torchx.Variable(u) )
      log.debug( "problem.step.logits: %s", yhat )
      log.debug( "problem.step.gs: %s", gs )
      p = fn.softmax( yhat, dim=1 ).squeeze()
      _, yhat = torch.max( p, dim=0 )
      log.debug( "problem.step.yhat: %s", yhat.item() )
      r = self.reward_model.reward( y, yhat.data )
    drain = self.energy_consumption( gs )
    self.battery_model.step( drain )
    if self.battery_model.energy == 0: # Used too much energy
      r = self.reward_model.reward( y, None )
    log.debug( "problem.step.r: %s", r )
    info = {"y": y.item(), "yhat": None if yhat is None else yhat.item(),
            "u": None if u is None else u.item(), "drain": drain}
    # Next state
    terminal = self._advance()
    xprime, _ = self._current
    return (xprime, self._tensor(r), self._tensor(terminal), info)

# ----------------------------------------------------------------------------
    
class SimpleImageClassificationProblem:
  def __init__( self, hardware_model, network, dataset,
                data_directory, reward_model, to_device, train,
                controller_macc=None, energy_cost_weight=0.5 ):
    self.hardware_model = hardware_model
    self.reward_model = reward_model
    self.network = network
    self.to_device = to_device
    self._train = train
    # self.t_subdivision = t_subdivision
    self.dataset = dataset
    data = dataset.load( root=data_directory, train=train )
    self.loader = DataLoader(
      data, batch_size=1, shuffle=True, pin_memory=True )
    self._data_itr = None
    self._current = None
    # flops() is currently only implemented for GatedDenseNet
    total_macc, gated_flops = self.network.flops( dataset.in_shape )
    gated_total = sum( c.macc for m in gated_flops for c in m )
    self.data_macc = total_macc
    self.ungated_data_macc = total_macc - gated_total
    self._controller_macc = controller_macc
    
    self.energy_cost_weight = energy_cost_weight
    # list of 1xC tensors containing macc count per gated component
    self.gated_macc = [
      self.to_device( torch.tensor( [c.macc for c in m] ).unsqueeze(0) )
      for m in gated_flops]
      
  @property
  def controller_macc( self ):
    return self._controller_macc
    
  @controller_macc.setter
  def controller_macc( self, x ):
    self._controller_macc = x
    self.max_energy = self.hardware_model.flop_energy(
      2*(self.data_macc + self.controller_macc) ) #.macc )
    
  def _advance( self ):
    try:
      x, y = next(self._data_itr)
      x = self.to_device( x )
      y = self.to_device( y )
      self._current = (x, y)
      return False
    except StopIteration:
      return True
  
  def _tensor( self, x ):
    x = [x] if np.isscalar( x ) else x
    t = torch.FloatTensor( x )
    t = t.unsqueeze(0)
    return self.to_device(t)
  
  def energy_consumption( self, gs ):
    # FIXME: Need to verify that this is giving the right answer
  
    # Elements of 'gs' are tuples; 2nd component contains data needed for
    # training the gate functions
    macc = self.controller_macc
    if gs is not None:
      gated_macc = [torch.sum(g[0].data * n, dim=1) 
                    for (g, n) in zip(gs, self.gated_macc)]
      gated_macc = sum( gated_macc )
      gated_macc += self.ungated_data_macc
      gated_macc = torch.sum( gated_macc ).item()
      macc += gated_macc
    nflop = 2*macc
    log.debug( "problem.energy_consumption.flops: %s", nflop )
    return self.hardware_model.flop_energy( nflop )
    
  def reset( self ):
    self._data_itr = iter(self.loader)
    terminal = self._advance()
    assert not terminal
    x, _y = self._current
    # Outputs from the data load are already tensors
    return x
      
  def step( self, u ):
    log.debug( "problem.step.u: %s %s", u, type(u) )
    x, y = self._current
    log.debug( "problem.step.x: %s", x )
    log.debug( "problem.step.x.size(): %s", x.size() )
    log.debug( "problem.step.y: %s", y.item() )
    if u is None: # Network turned off
      log.debug( "problem.step.yhat: None" )
      yhat = None
      r = self.reward_model.reward( y, yhat )
      gs = None
    else:
      u = self.to_device( torch.tensor( [float(u)] ) )
      # u = self._tensor( u )
      yhat, gs = self.network( torchx.Variable(x), torchx.Variable(u) )
      log.debug( "problem.step.logits: %s", yhat )
      log.debug( "problem.step.gs: %s", gs )
      p = fn.softmax( yhat, dim=1 ).squeeze()
      _, yhat = torch.max( p, dim=0 )
      log.debug( "problem.step.yhat: %s", yhat.item() )
      r = self.reward_model.reward( y, yhat.data )
    # Changed for 'simple' problem
    drain = self.energy_consumption( gs )
    prop_drain = drain / self.max_energy
    log.debug( "problem.step.drain: %s (%s)", drain, prop_drain )
    r -= self.energy_cost_weight * prop_drain
      
    log.debug( "problem.step.r: %s", r )
    info = {"y": y.item(), "yhat": None if yhat is None else yhat.item(),
            "u": None if u is None else u.item(), "drain": drain}
    # Next state
    terminal = self._advance()
    xprime, _ = self._current
    return (xprime, self._tensor(r), self._tensor(terminal), info)
    
# ----------------------------------------------------------------------------
    
class EnergyCostImageClassificationProblem:
  def __init__( self, energy_cost, hardware_model, network, dataset,
                data_directory, reward_model, to_device, train,
                controller_macc=None, energy_cost_weight=0.5 ):
    self.energy_cost = energy_cost
    self.hardware_model = hardware_model
    self.reward_model = reward_model
    self.network = network
    self.to_device = to_device
    self._train = train
    # self.t_subdivision = t_subdivision
    self.dataset = dataset
    data = dataset.load( root=data_directory, train=train )
    self.loader = DataLoader(
      data, batch_size=1, shuffle=True, pin_memory=True )
    self._data_itr = None
    self._current = None
    # flops() is currently only implemented for GatedDenseNet
    total_macc, gated_flops = self.network.flops( dataset.in_shape )
    gated_total = sum( c.macc for m in gated_flops for c in m )
    self.data_macc = total_macc
    self.ungated_data_macc = total_macc - gated_total
    self._controller_macc = controller_macc
    
    self.energy_cost_weight = energy_cost_weight
    # list of 1xC tensors containing macc count per gated component
    self.gated_macc = [
      self.to_device( torch.tensor( [c.macc for c in m] ).unsqueeze(0) )
      for m in gated_flops]
      
  @property
  def controller_macc( self ):
    return self._controller_macc
    
  @controller_macc.setter
  def controller_macc( self, x ):
    self._controller_macc = x
    self.max_energy = self.hardware_model.flop_energy(
      2*(self.data_macc + self.controller_macc) ) #.macc )
    
  def _advance( self ):
    try:
      x, y = next(self._data_itr)
      x = self.to_device( x )
      y = self.to_device( y )
      self._current = (x, y)
      return False
    except StopIteration:
      return True
  
  def _tensor( self, x ):
    x = [x] if np.isscalar( x ) else x
    t = torch.FloatTensor( x )
    t = t.unsqueeze(0)
    return self.to_device(t)
  
  def energy_consumption( self, gs ):
    # FIXME: Need to verify that this is giving the right answer
  
    # Elements of 'gs' are tuples; 2nd component contains data needed for
    # training the gate functions
    macc = self.controller_macc
    if gs is not None:
      gated_macc = [torch.sum(g[0].data * n, dim=1) 
                    for (g, n) in zip(gs, self.gated_macc)]
      gated_macc = sum( gated_macc )
      gated_macc += self.ungated_data_macc
      gated_macc = torch.sum( gated_macc ).item()
      macc += gated_macc
    nflop = 2*macc
    log.debug( "problem.energy_consumption.flops: %s", nflop )
    return self.hardware_model.flop_energy( nflop )
    
  def reset( self ):
    self.energy_cost.reset()
    self._data_itr = iter(self.loader)
    terminal = self._advance()
    assert not terminal
    x, _y = self._current
    # Outputs from the data load are already tensors
    return x
      
  def step( self, u ):
    log.debug( "problem.step.u: %s %s", u, type(u) )
    x, y = self._current
    log.debug( "problem.step.x: %s", x )
    log.debug( "problem.step.x.size(): %s", x.size() )
    log.debug( "problem.step.y: %s", y.item() )
    if u is None: # Network turned off
      log.debug( "problem.step.yhat: None" )
      yhat = None
      r = self.reward_model.reward( y, yhat )
      gs = None
    else:
      u = self.to_device( torch.tensor( [float(u)] ) )
      # u = self._tensor( u )
      yhat, gs = self.network( torchx.Variable(x), torchx.Variable(u) )
      log.debug( "problem.step.logits: %s", yhat )
      log.debug( "problem.step.gs: %s", gs )
      p = fn.softmax( yhat, dim=1 ).squeeze()
      _, yhat = torch.max( p, dim=0 )
      log.debug( "problem.step.yhat: %s", yhat.item() )
      r = self.reward_model.reward( y, yhat.data )
    # Changed for 'simple' problem
    drain = self.energy_consumption( gs )
    prop_drain = drain / self.max_energy
    log.debug( "problem.step.drain: %s (%s)", drain, prop_drain )
    log.debug( "problem.step.energy_cost: %s", self.energy_cost.cloudiness )
    r -= (0.1 + self.energy_cost.cloudiness) * self.energy_cost_weight * prop_drain
    
    # rmin = self.reward_model.Rmin() - 1.1*self.energy_cost_weight*1.0
    # rmax = self.reward_model.Rmax()
    # r = (r - rmin) / (rmax - rmin)
      
    log.debug( "problem.step.r: %s", r )
    info = {"y": y.item(), "yhat": None if yhat is None else yhat.item(),
            "u": None if u is None else u.item(), "drain": drain}
    # Next state
    terminal = self._advance()
    xprime, _ = self._current
    self.energy_cost.step()
    return (xprime, self._tensor(r), self._tensor(terminal), info)
    
# ----------------------------------------------------------------------------
    
class ContinuousActionModel:
  @property
  def action_space( self ):
    # Action: (enable network? (2-dim softmax), target utilization)
    return gym.spaces.Box(
      low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32 )
    # FIXME: The action space is actually a hybrid discrete and continuous
    # space, but DDPG can't handle that. It would be better to use an adapter
    # to flatten the space rather than hard-coding the wrong representation.
    # self.action_space = gym.spaces.Tuple([
      # gym.spaces.Discrete( 2 ),
      # gym.spaces.Box( low=0, high=1, shape=(1,), dtype=np.float32 ) ])
      
  def action_to_control( self, a ):
    enable = a[1] > a[0]
    u = a[2]
    if enable == 0:
      u = None
    return u
      
class DiscreteActionModel:
  def __init__( self, ngate_levels, include_none=True ):
    assert ngate_levels >= 1
    if ngate_levels == 1:
      self._us = [1.0]
    else:
      inc = 1.0 / (ngate_levels - 1)
      self._us = [i * inc for i in range(ngate_levels)]
    log.debug( "DiscreteActionModel.us: %s", self._us )
    self._include_none = include_none
    self._nactions = ngate_levels + (1 if include_none else 0)
    
  @property
  def action_space( self ):
    return gym.spaces.Discrete( self._nactions )
    
  def action_to_control( self, a ):
    assert isinstance(a, int)
    if a == len(self._us):
      assert self._include_none
      return None
    else:
      return self._us[a]

# ----------------------------------------------------------------------------
      
class SolarPoweredGatedNetworkEnv(gym.Env):
  def __init__( self, problem, action_model, to_device ):
    self.problem = problem
    self.action_model = action_model
    self.to_device = to_device
    # States: (time step, charge, illumination history) x (image shape)
    # Time step is encoded as (sin, cos) of normalized time of day (midnight=0)
    # Illumination and Charge are normalized to [0,1]
    system = gym.spaces.Box( low=np.array( [-1, -1, 0, 0, 0, 0]),
                             high=np.array([ 1,  1, 1, 1, 1, 1]),
                             dtype=np.float32 )
    img = gym.spaces.Box(
      low=0, high=1, shape=self.problem.dataset.in_shape, dtype=np.float32 )
    self.observation_space = gym.spaces.Tuple([system, img])
  
  @property
  def action_space( self ):
    return self.action_model.action_space
  
  @property
  def t( self ):
    return self.problem.battery_model.exposure_model.t
    
  @property
  def T( self ):
    return self.problem.battery_model.exposure_model.T
  
  @property
  def exposure( self ):
    return self.problem.battery_model.exposure_model.exposure
  
  def _system_state( self ):
    self.hourly_exposure( self.exposure )
    self.daily_exposure( self.exposure )
    ts = math.sin( 2*math.pi * (self.t / self.T) )
    tc = math.cos( 2*math.pi * (self.t / self.T) )
    charge = self.problem.battery_model.charge
    s0 = self.exposure
    s1 = self.hourly_exposure.mean()
    s2 = self.daily_exposure.mean()
    return np.array([ts, tc, charge, s0, s1, s2])
  
  def _full_state( self, ximg ):
    xsys = self._system_state()
    xsys = torch.FloatTensor( [xsys] )
    xsys = self.to_device( xsys )
    return torchx.TensorTuple( [xsys, ximg] )
  
  def reset( self ):
    self.hourly_exposure = MovingMeanAccumulator( 60 )
    self.daily_exposure = MovingMeanAccumulator( 60 * 24 )
    ximg = self.problem.reset()
    return self._full_state( ximg )
    
  def step( self, a ):
    u = self.action_model.action_to_control( a )
    ximg, r, terminal, info = self.problem.step( u )
    info["a"] = a
    x = self._full_state( ximg )
    return (x, r, terminal, info)

# ----------------------------------------------------------------------------
    
class GatedNetworkEnv(gym.Env):
  def __init__( self, problem, action_model, to_device ):
    self.problem = problem
    self.action_model = action_model
    self.to_device = to_device
    img = gym.spaces.Box(
      low=0, high=1, shape=self.problem.dataset.in_shape, dtype=np.float32 )
    self.observation_space = img
  
  @property
  def action_space( self ):
    return self.action_model.action_space
  
  def _full_state( self, ximg ):
    return ximg
  
  def reset( self ):
    ximg = self.problem.reset()
    return self._full_state( ximg )
    
  def step( self, a ):
    u = self.action_model.action_to_control( a )
    ximg, r, terminal, info = self.problem.step( u )
    info["a"] = a
    return (ximg, r, terminal, info)

# ----------------------------------------------------------------------------
      
class EnergyCostGatedNetworkEnv(gym.Env):
  def __init__( self, problem, action_model, to_device ):
    self.problem = problem
    self.action_model = action_model
    self.to_device = to_device
    # States: (energy_cost)
    system = gym.spaces.Box( low=np.array( [self.problem.energy_cost.minval]),
                             high=np.array([self.problem.energy_cost.maxval]),
                             dtype=np.float32 )
    img = gym.spaces.Box(
      low=0, high=1, shape=self.problem.dataset.in_shape, dtype=np.float32 )
    self.observation_space = gym.spaces.Tuple([system, img])
  
  @property
  def action_space( self ):
    return self.action_model.action_space
  
  def _system_state( self ):
    # FIXME: Make this name more generic
    return np.array([self.problem.energy_cost.cloudiness])
  
  def _full_state( self, ximg ):
    xsys = self._system_state()
    xsys = torch.FloatTensor( [xsys] )
    xsys = self.to_device( xsys )
    return torchx.TensorTuple( [xsys, ximg] )
  
  def reset( self ):
    ximg = self.problem.reset()
    return self._full_state( ximg )
    
  def step( self, a ):
    u = self.action_model.action_to_control( a )
    ximg, r, terminal, info = self.problem.step( u )
    info["a"] = a
    x = self._full_state( ximg )
    return (x, r, terminal, info)
    
# ----------------------------------------------------------------------------
    
class SolarEpisodeLogger(rl.EpisodeObserver):
  def __init__( self, log, level=logging.INFO, prefix="solar." ):
    self.log = log
    self.level = level
    self.prefix = prefix
    
  def begin( self ):
    self.log.log( self.level, "%sbegin", self.prefix )
    
  def end( self ):
    self.log.log( self.level, "%send", self.prefix )
    
  def observation( self, x ):
    if isinstance(x, tuple):
      xsys = x[0]
      self.log.log( self.level, "%sstate: %s", self.prefix,
        "[{}]".format(", ".join(["{:0.4f}".format(x) for x in xsys.tolist()[0]])))
    
  def action( self, a ):
    self._a = a
    
  def reward( self, r ):
    self._r = r.item()
    
  def info( self, info ):
    d = info.copy()
    d["r"] = self._r
    d["drain"] = "{:0.4f}".format(d["drain"])
    log.info( "solar.action: %s", d )    
    
# ----------------------------------------------------------------------------

def main():
  import random
  import statistics
  rng = random.Random( 6991 )
  sunlight = SinusoidSunlightModel( rng )
  clouds = RandomWalkCloudModel( rng )
  exposure = ExposureModel( sunlight, clouds )
  battery = SolarBatteryModel(
    rng, exposure, capacity_J=20, max_generation_mW=1 )
  hardware = TegraK1()
  
  
  # problem = ImageClassificationProblem(
    # battery, hardware, self.network,
    # self.dataset, self.args.data_directory, train )
  
  
  exposure.reset()
  hist = [0] * 10
  mean = 0
  N = 365 * 60*24 #exposure.T
  transitions = 0
  ex = 0
  sun = 0
  for t in range(N):
    ex += exposure.exposure
    sun += exposure.sunlight_model.intensity
    # s = exposure.cloud_model.hidden
    i = int(math.floor(exposure.cloud_model.cloudiness * 10))
    i = min(i, 9)
    hist[i] += 1
    mean += (exposure.cloud_model.cloudiness - mean) / (t+1)
    # print( "sunny" if clouds.hidden == 0 else "rainy" )
    # if exposure.t == 0:
      # print( "----------" )
    # print( "cloudiness: {}".format( exposure.cloudiness ) )
    # print( "exposure: {}; sunlight: {}; cloudiness: {}".format(
           # exposure.exposure, exposure.sunlight_model.intensity, exposure.cloudiness ))
    exposure.step()
    # if exposure.cloud_model.hidden != s:
      # transitions += 1
  for i in range(len(hist)):
    print( "{}: {}".format(i, hist[i]) )
  print( "mean: {}".format( mean ) )
  print( "transitions: {}".format( transitions ) )
  print( "sun hours per day: {}".format( sun / 365 / 60 ) )
  print( "exposure hours per day: {}".format( ex / 365 / 60 ) )
  
  exs = []
  for n in range(5):
    ex = 0
    exposure.reset()
    # for d in range(7):
    for t in range(7*60*24):
      ex += (exposure.exposure - ex) / (t+1)
      exposure.step()
    exs.append( ex )
      # print( 24*ex )
  print( "mean exposure: {}".format(statistics.mean(exs)) )
  print( "stdev exposure: {}".format(statistics.stdev(exs)) )
  
  # exposure.reset()
  # ex = []
  # for t in range(24*60):
    # ex.append( exposure.cloud_model.cloudiness )
    # exposure.step()
  # print( ex )
  # print( statistics.mean( ex ) )
  # print( statistics.stdev( ex ) )
  
if __name__ == "__main__":
  main()
    