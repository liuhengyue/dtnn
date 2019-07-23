import argparse
import contextlib
from   functools import reduce
import glob
import logging
import math
import multiprocessing
import operator
import os
import random
import re
import shlex
import shutil
import sys

import numpy as np
import numpy.random

import torch
import torchvision
import torchvision.transforms as transforms

from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.init as init
from   torch.nn.parameter import Parameter
import torch.optim as optim

import nnsearch.app.impl.gated as impl
from   nnsearch.app.impl.gated import GatedNetworkApp
import nnsearch.argparse as myarg
import nnsearch.logging as mylog
from   nnsearch.pytorch import rl
import nnsearch.pytorch.act as act
from   nnsearch.pytorch.data import datasets
import nnsearch.pytorch.gated.control.direct as direct
import nnsearch.pytorch.gated.densenet as densenet
import nnsearch.pytorch.gated.learner as glearner
import nnsearch.pytorch.gated.module as gated_module
import nnsearch.pytorch.gated.resnext as resnext
import nnsearch.pytorch.gated.strategy as strategy
import nnsearch.pytorch.gated.vgg as vgg
from   nnsearch.pytorch.models import resnet
import nnsearch.pytorch.modules as modules
from   nnsearch.pytorch.modules import FullyConnected, GlobalAvgPool2d
import nnsearch.pytorch.parameter as parameter
from   nnsearch.pytorch.rl import bandit
from   nnsearch.pytorch.rl import ddpg
from   nnsearch.pytorch.rl import dqn
from   nnsearch.pytorch.rl import gymx
from   nnsearch.pytorch.rl import policy
from   nnsearch.pytorch.rl import replay
import nnsearch.pytorch.rl.env.solar as solar
from   nnsearch.pytorch.rl.random_process import GaussianWhiteNoiseProcess
import nnsearch.pytorch.torchx as torchx
from   nnsearch.statistics import MeanAccumulator, MovingMeanAccumulator

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

class Bunch:
  def __init__( self, **kwargs ):
    self.__dict__.update( kwargs )
    
class PrintWeights:
  def __call__( self, module ):
    if isinstance( module, (torch.nn.Conv2d, torch.nn.Linear) ):
      log.verbose( module )
      log.verbose( module.weight.data )
      
class PrintGradients:
  def __call__( self, module ):
    log.micro( "grad: %s", module )
    if isinstance( module, (torch.nn.Conv2d, torch.nn.Linear) ):
      if module.weight.grad is not None:
        log.micro( "grad:\n%s", module.weight.grad.data )
      else:
        log.micro( "grad: None" )

# ----------------------------------------------------------------------------
# Cmd parsing

class MyArgumentParser(argparse.ArgumentParser):
  def convert_arg_line_to_args( self, arg_line ):
    return shlex.split( arg_line )
        
def parse_schedule():

  class ScheduleParser(argparse.Action):
    def __init__( self, *args, **kwargs ):
      super().__init__( *args, **kwargs )

    def __call__( self, parser, namespace, values, option_string=None ):
      if values is None:
        setattr(namespace, self.dest, None)
        return
      
      tokens = values.split( "," )
      if tokens[0] == "constant":
        p = parameter.Constant( option_string, float(tokens[1]) )
      elif tokens[0] == "linear":
        p = parameter.LinearSchedule( option_string, *map(float, tokens[1:]) )
      elif tokens[0] == "geometric":
        p = parameter.GeometricSchedule( option_string, *map(float, tokens[1:]) )
      elif tokens[0] == "harmonic":
        p = parameter.HarmonicSchedule( option_string, *map(float, tokens[1:]) )
      elif tokens[0] == "step":
        xs = [float(r) for r in tokens[1::2]]
        times = [int(t) for t in tokens[2::2]]
        p = parameter.StepSchedule( option_string, xs, times )
      elif tokens[0] == "cosine":
        p = parameter.CosineSchedule( option_string,
          float(tokens[1]), float(tokens[2]), int(tokens[3]), int(tokens[4]) )
      else:
        parser.error( option_string )
      setattr(namespace, self.dest, p)
      
  return ScheduleParser
  
def create_cmd_parser():
  parser = MyArgumentParser( description="Training gated neural networks", 
    fromfile_prefix_chars="@", allow_abbrev=False )
  return parser
    
# ----------------------------------------------------------------------------

# def remove_densenet_classifier( network ):
  # assert isinstance(network.fn, nn.ModuleList)
  
  # # Drop the final classification layer
  # fn_layers = [m for m in network.fn] # ModuleList doesn't support slicing
  # fn_layers = fn_layers[:-1]
  # assert isinstance(fn_layers[-1], GlobalAvgPool2d)
  
  # # Not sure if sharing a ModuleList is OK
  # gate_layers = [m for m in network.gate.gate_modules]
  # gated_modules = [m for m in fn_layers 
                   # if isinstance(m, densenet.GatedDenseNetBlock)]
  # gate = strategy.SequentialGate( gate_layers )
  # return gated_module.GatedChainNetwork( gate, fn_layers, gated_modules )

class ParameterizedActionOutput(nn.Module):
  def __init__( self, nactions, nparams ):
    super().__init__()
    self.nactions = nactions
    self.nparams = nparams
    
  def forward( self, x ):
    a = x.narrow(1, 0, self.nactions)
    p = x.narrow(1, self.nactions, self.nparams)
    a = fn.softmax( a, dim=1 )
    p = modules.clamp_invert_gradient(
      p, min=torch.tensor(0.0), max=torch.tensor(1.0) )
    return (a, p)
  
class BlindSolarActor(nn.Module):
  def __init__( self, observation_space, action_space, nhidden ):
    super().__init__()
    sysdim = observation_space.spaces[0].shape[0]
    # FIXME: Hardcoded action space from 'solar'
    nactions = 2
    nparams = 1
    self.fn = nn.Sequential(
      nn.Linear(sysdim, nhidden), nn.ReLU(),
      nn.Linear(nhidden, nactions+nparams),
      ParameterizedActionOutput(nactions, nparams))
  
  def forward( self, s ):
    xsys, ximg = s
    a, p = self.fn( xsys )
    return torch.cat( [a, p], dim=1 )
    # _, a = torch.max( a, dim=1, keepdim=True )
    # return (a, p)

class BlindSolarCritic(nn.Module):
  def __init__( self, observation_space, action_space, nhidden ):
    super().__init__()
    sysdim = observation_space.spaces[0].shape[0]
    # FIXME: Hardcoded action space from 'solar'
    nactions = 2
    nparams = 1
    self.head = nn.Sequential( nn.Linear(sysdim, nhidden), nn.ReLU() )
    self.tail = nn.Sequential( nn.Linear(nhidden + nactions+nparams, 1) )
    
  def forward( self, s, a ):
    xsys, ximg = s
    s = self.head( xsys )
    x = self.tail( torch.cat( [s, a], dim=1 ) )
    return x
    
# ----------------------------------------------------------------------------
    
class ContextualQNetwork(nn.Module):
  def __init__( self, observation_space, action_space,
                context_network, ctx_out_channels, nhidden ):
    super().__init__()
    self.sysdim = observation_space.spaces[0].shape[0]
    self.ctx_out_channels = ctx_out_channels
    # FIXME: Hardcoded action space from 'solar'
    nactions = action_space.n
    self.features = context_network
    self.q = nn.Sequential(
      nn.Linear( self.sysdim + self.ctx_out_channels, nhidden, bias=False ),
      nn.ReLU(),
      nn.Linear( nhidden, nactions, bias=False ) )
    
  def forward( self, s ):
    xsys, ximg = s
    f = self.features( ximg )
    x = torch.cat( [f, xsys], dim=1 )
    return self.q( x )

@torchx.flops.register(ContextualQNetwork)
def _( layer, in_shape ):
  f1 = torchx.flops( layer.features, in_shape )
  f2 = torchx.flops( layer.q, [layer.sysdim + layer.ctx_out_channels] )
  # FIXME: This is wrong if we add fields to Flops tuple
  return torchx.Flops( f1.macc + f2.macc )
    
# ----------------------------------------------------------------------------
    
class SolarExplorePolicy(rl.Policy):
  def __init__( self, base, action_space, explore_process ):
    self.pi = base
    self.action_space = action_space
    self.explore_process = explore_process
    
  def reset( self ):
    self.pi.reset()
    self.explore_process.reset()

  def observe( self, s ):
    pi_s = self.pi.observe( s )
    def action( rng ):
      # FIXME: Need to add exploration to the discrete action choice. Easiest
      # thing is to add noise to the softmax values, but then you need to
      # re-normalize them.
      delta = self.explore_process.sample()
      log.debug( "explore.delta: %s", delta )
      flat = pi_s( rng )
      log.debug( "explore.flat: %s", flat )
      flat[2] += delta
      log.debug( "explore.randomized: %s", flat )
      flat = np.clip( flat, self.action_space.low, self.action_space.high )
      log.debug( "explore.clipped: %s", flat )
      return flat
      
      # a = flat[:2]
      # a = np.argmax( a )
      # p = flat[2]
      # plow, phigh = self.action_space.low[1], self.action_space.high[1]
      # p = np.clip( p + delta, plow, phigh )
      # return np.array( [a, p], dtype=np.float32 )
      
      # return a, p
    return action

# ----------------------------------------------------------------------------

class App(GatedNetworkApp):
  def __init__( self ):
    parser = MyArgumentParser( description="RL control of gated networks", 
      fromfile_prefix_chars="@", allow_abbrev=False )
    super().__init__( parser )
    
    # Parameters that vary over time
    self.hyperparameters = [v for v in vars(self.args).values() 
                            if isinstance(v, parameter.Hyperparameter)]
    
    # Logger setup
    # FIXME: Adding log levels should happen at package-level
    mylog.add_log_level( "MICRO",   logging.DEBUG - 5 )
    mylog.add_log_level( "NANO",    logging.DEBUG - 6 )
    mylog.add_log_level( "PICO",    logging.DEBUG - 7 )
    mylog.add_log_level( "FEMTO",   logging.DEBUG - 8 )
    mylog.add_log_level( "VERBOSE", logging.INFO - 5 )
    root_logger = logging.getLogger()
    root_logger.setLevel( mylog.log_level_from_string( self.args.log_level ) )
    # Need to set encoding or Windows will choke on ellipsis character in
    # PyTorch tensor formatting
    handler = logging.FileHandler(
      os.path.join( self.args.output, "iclr.log" ), "w", "utf-8")
    # handler.setFormatter( logging.Formatter("%(levelname)s:%(name)s: %(message)s") )
    handler.setFormatter( logging.Formatter("%(levelname)s: %(message)s") )
    root_logger.addHandler(handler)
    # logging.basicConfig()
    # logging.getLogger().setLevel( logging.INFO )
    
    log.info( "Git revision: %s", mylog.git_revision() )
    log.info( self.args )
    
    self.master_rng = random.Random( self.args.seed )
    self.train_rng = random.Random()
    self.eval_rng  = random.Random()
    def next_seed( seed=None ):
      if seed is None:
        seed = self.master_rng.randrange( 2**31 - 1 )
      random.seed( seed )
      self.train_rng.seed( seed+10 )
      self.eval_rng.seed( seed+15 )
      numpy.random.seed( seed+20 )
      torch.manual_seed( seed+30 )
      return seed  
    seed = next_seed()
    log.info( "==== Initializing: seed=%s", seed )
    
    super().init( self.args )
    
    if self.args.gpu:
      self.device = torch.device( "cuda" )
      # self.to_device = lambda t: t.cuda()
      self.to_device = lambda t: t.to( self.device )
    else:
      self.device = torch.device( "cpu" )
      self.to_device = lambda t: t
      
    self.init_dataset( self.args )
    
    self.start_epoch = 0 # Might get overwritten if loading checkpoint
    self.init_data_network()
    
    self.train_env = self.make_environment( self.train_rng, train=True )
    self.eval_env  = self.make_environment( self.eval_rng, train=False )
    self.observation_space = self.train_env.observation_space
    self.action_space      = self.train_env.action_space
    
    self.controller_macc = self.init_learner()
    # FIXME: Don't like setting these after env is already constructed, but at
    # present there is an initialization cycle that we have to break
    self.train_env.problem.controller_macc = self.controller_macc
    self.eval_env.problem.controller_macc = self.controller_macc
    
    total, gated = self.data_network.flops( self.dataset.in_shape )
    gtotal = sum( c.macc for m in gated for c in m )
    print( "TOTAL", total )
    print( gtotal )
    print( total - gtotal )
  
  def _install_arguments( self, parser ):
    super()._install_arguments( parser )
    
    rl_group = parser.add_argument_group( "rl" )
    rl_group.add_argument( "--env", type=str,
      choices=["solar", "simple", "cost"], default="solar", help="Environment" )
    rl_group.add_argument( "--agent", type=str,
      choices=["ddpg", "dqn", "double_dqn", "random", "reinforce_bandit"], help="Agent algorithm" )
    rl_group.add_argument( "--discount", type=float,
      action=myarg.range_check(0, 1), default=1.0, help="Discount factor" )
    rl_group.add_argument( "--explore", type=str,
      default="epsilon_greedy,constant,0.5",
      help="Exploration policy." )
    rl_group.add_argument( "--train-episodes", type=int, default=0,
      help="Number of training episodes." )
    rl_group.add_argument( "--train-episode-length", type=int, default=None,
      help="Maximum length of each training episode (default: unlimited)." )
    rl_group.add_argument( "--eval-episodes", type=int, default=0,
      help="Number of training episodes." )
    rl_group.add_argument( "--eval-episode-length", type=int, default=None,
      help="Maximum length of each training episode (default: unlimited)." )
    rl_group.add_argument( "--replay", type=str, default="circular,100",
      help="Experience replay method" )
    rl_group.add_argument( "--replay-priority", type=str,
      choices=["error", "return"], default=None,
      help="Method for determining replay priority" )
    replay_init_group = rl_group.add_mutually_exclusive_group()
    replay_init_group.add_argument( "--replay-init-size", type=int,
      default=None, action=myarg.range_check(0, None),
      help="Number of random samples to initialize replay memory." )
    replay_init_group.add_argument( "--replay-init-proportion", type=float,
      default=None, action=myarg.range_check(0, 1),
      help="Proportion of replay memory to initialize." )
    rl_group.add_argument( "--update-interval", type=int, default=1,
      help="Number of \"steps\" between learning updates" )
    rl_group.add_argument( "--target-update-interval", type=int, default=100,
      help="Number of \"steps\" between target network updates" )
    rl_group.add_argument( "--action-model", type=str, default="discrete,17",
      help="Action model for RL controller" )
    rl_group.add_argument( "--reward-model", type=str, default="uniform",
      help="Reward model for the classification problem." )
    rl_group.add_argument( "--reward-fp", type=float, default=None,
      help="Reward for false positives (if using 'target' reward model." )
    rl_group.add_argument( "--reward-fn", type=float, default=None,
      help="Reward for false negatives (if using 'target' reward model." )
    rl_group.add_argument( "--representation", type=str, default=None,
      choices=["blind", "blockdrop"], help="Representation of images" )
    rl_group.add_argument( "--load-feature-network", type=str, default=None,
      help="File containing pretrained feature network for controller" )
    rl_group.add_argument( "--freeze-feature-network", action="store_true",
      help="Disable learning of feature network." )
    rl_group.add_argument( "--load-data-network", type=str, default=None,
      help="File containing pretrained gated data network." )
    rl_group.add_argument( "--freeze-data-network", action="store_true",
      help="Disable learning of feature network." )
    rl_group.add_argument( "--explore-temperature", default="constant,1",
      type=parameter.schedule_spec("--explore-temperature"),
      help="Temperature schedule for applicable exploration methods" )
    env_group = parser.add_argument_group( "environment" )
    env_group.add_argument( "--environment", type=str, default=None,
      help="String containing Python code that instantiates an environment."
        " Executes in the context of 'import epsilon.rl.env.multitask as mt'." )
    env_group.add_argument( "--train-task-schedule", type=str, default="uniform",
      help="Schedule for choosing which task each training episode comes from." )
    env_group.add_argument( "--eval-task-schedule", type=str, default="uniform",
      help="Schedule for choosing which task each eval episode comes from." )
    solar_group = parser.add_argument_group( "solar" )
    solar_group.add_argument( "--solar-battery-capacity-J", type=float, default=10,
      help="Capacity of battery in (J)" )
    solar_group.add_argument( "--solar-max-generation-mW", type=float, default=0.6,
      help="Power generated at full sunlight exposure (mW)" )
    solar_group.add_argument( "--energy-cost", type=str,
      choices=["uniform", "random_walk"], default="uniform",
      help="Energy cost model" )
    solar_group.add_argument( "--energy-cost-weight", type=float, default=0.7,
      help="Weight of energy cost in reward calculation" )
    solar_group.add_argument( "--solar-policy-nhidden", type=int, default=128,
      help="Number of hidden units in FC part of policy network" )
  
  def make_reward_model( self ):
    tokens = self.args.reward_model.split( "," )
    model_type = tokens[0]
    if model_type == "target":
      target_class = int(tokens[1])
      if self.args.reward_fn is None:
        self.parser.error( "--reward-model=target requires --reward-fn" )
      if self.args.reward_fp is None:
        self.parser.error( "--reward-model=target requires --reward-fp" )
      confusion_matrix = np.eye( self.dataset.nclasses ) - 1.0
      for j in range(self.dataset.nclasses):
        if j != target_class:
          confusion_matrix[target_class][j] = self.args.reward_fn
          confusion_matrix[j][target_class] = self.args.reward_fp
      none_prediction = -np.ones( self.dataset.nclasses )
      none_prediction[target_class] = self.args.reward_fn
      log.info( "reward_model.confusion_matrix:\n%s", confusion_matrix )
      log.info( "reward_model.none_prediction: %s", none_prediction )
      return solar.ConfusionRewardModel( confusion_matrix, none_prediction )
    elif model_type == "uniform":
      return solar.UniformRewardModel()
    # else:
    self.parser.error( "--reward-model={}".format(self.args.reward_model) )
    
  def make_action_model( self ):
    tokens = self.args.action_model.split( "," )
    model_type = tokens[0]
    if model_type == "discrete":
      ngate_levels = int(tokens[1])
      return solar.DiscreteActionModel( ngate_levels )
    elif model_type == "continuous":
      return solar.ContinuousActionModel()
    # else:
    self.parser.error( "--action-model={}".format(self.args.action_model) )
  
  def make_environment( self, rng, train ):
    # The sunlight model gives 7.64 hours of max intensity-equivalent
    # sunlight per day
    sunlight = solar.SinusoidSunlightModel( rng )
    # The standard cloud model averages about 0.503 cloudiness
    clouds = solar.RandomWalkCloudModel( rng )
    # The exposure model averages 4.23 hours of max intensity-equivalent
    # exposure per day
    exposure = solar.ExposureModel( sunlight, clouds )
    # The DenseNet model with blockdrop controller network uses 23.45 J / day
    # at u = 1 and 1 image / minute (1440 images / day)
    # With max_generation = 1mW, we generate 15.23 J / day on average
    # That means we can sustain 0.01058 J / image on average
    #
    # For the DenseNet model,
    # u = None  : 0.0010 J / image
    # u = 0.0   : 0.0041 J / image
    # u = 0.0625: 0.0048 J / image
    # u = 0.125 : 0.0056 J / image
    # u = 0.1875: 0.006350 J / image
    # u = 0.25  : 0.007114 J / image
    # u = 0.5   : 0.01017 J / image
    # u = 0.5625: 0.01094 J / image
    # u = 0.75  : 0.01323 J / image
    # u = 1.0   : 0.0163 J / image
    
    # Generation of 0.6 mW means we can sustain 0.006348 J / image
    battery = solar.SolarBatteryModel(
      rng, exposure, capacity_J=self.args.solar_battery_capacity_J,
      max_generation_mW=self.args.solar_max_generation_mW )
    hardware = solar.TegraK1()
    reward = self.make_reward_model()
    actions = self.make_action_model()
    if self.args.env == "solar":
      problem = solar.ImageClassificationProblem(
        battery, hardware, self.data_network,
        self.dataset, self.args.data_directory, reward, self.to_device, train )
      return solar.SolarPoweredGatedNetworkEnv( problem, actions, self.to_device )
    elif self.args.env == "simple":
      problem = solar.SimpleImageClassificationProblem(
        hardware, self.data_network,
        self.dataset, self.args.data_directory, reward, self.to_device, train,
        energy_cost_weight=self.args.energy_cost_weight )
      return solar.GatedNetworkEnv( problem, actions, self.to_device )
    elif self.args.env == "cost":
      # FIXME: Give this model a more generic name
      if self.args.energy_cost == "random_walk":
        energy_cost = solar.RandomWalkCloudModel( rng, minval=0.1, maxval=1.0 )
      elif self.args.energy_cost == "uniform":
        energy_cost = solar.UniformCloudModel( rng, minval=0.1, maxval=1.0 )
      problem = solar.EnergyCostImageClassificationProblem(
        energy_cost, hardware, self.data_network,
        self.dataset, self.args.data_directory, reward, self.to_device, train,
        energy_cost_weight=self.args.energy_cost_weight )
      return solar.EnergyCostGatedNetworkEnv( problem, actions, self.to_device )
    else:
      self.parser.error( "--env={}".format(self.args.env) )
    
  def make_replay( self ):
    tokens = self.args.replay.split( "," )
    if tokens[0] == "circular":
      replay_size = int(tokens[1])
      replay_buffer = replay.CircularList( replay_size )
      # FIXME: Implement replay initialization
      # fill_replay( rng, env, replay_buffer, replay_size )
      return replay_buffer
    # else:
    self.parser.error( "--replay={}".format(self.args.replay) )
  
  def init_network_parameters( self, network, from_file=None ):
    if from_file is not None:
      self.checkpoint_mgr.load_parameters(
        from_file, network, strict=self.args.strict_load )
    else:
      # Initialize weights
      # FIXME: make `initialize_weights` a method of a superclass
      network.apply( impl.initialize_weights(self.args) )
  
  def init_data_network( self ):
    self.data_network = self.init_gated_network_structure()
    log.info( "data_network:\n%s", self.data_network )
    if self.args.freeze_data_network:
      torchx.freeze( self.data_network )
      self.data_network = modules.FrozenBatchNorm( self.data_network )
    # Load or initialize parameters
    if (self.args.load_checkpoint is not None 
        and self.args.load_data_network is not None):
      self.parser.error( "--load-checkpoint and --load-feature-network are"
                         " mutually exclusive" )
    from_file = self.args.load_data_network
    if self.args.load_checkpoint is not None:
      from_file = self.checkpoint_mgr.get_checkpoint_file( 
        "data_network", self.args.load_checkpoint )
      self.start_epoch = self.checkpoint_mgr.epoch_of_model_file( from_file )
    self.init_gated_network_parameters( self.data_network, from_file )
    self.data_network.to( self.device )
    
  def debug( self ):
    # testset     = self.dataset.load( root=self.args.data_directory, train=False )
    # testloader  = torch.utils.data.DataLoader( testset,
      # batch_size=self.args.batch, shuffle=False, pin_memory=True,
      # num_workers=self.args.data_workers )
    
    env = self.eval_env
    learner = glearner.GatedNetworkLearner(
      # self.data_network,
      env.problem.network,
      self.make_optimizer(self.data_network.parameters()),
      self.args.learning_rate, self.data_network.gate, impl.gate_control(self.args) )
    
    elapsed_epochs = 0
    # Hyperparameters interpret their 'epoch' argument as index of the current
    # epoch; we want the same hyperparameters as in the most recent training
    # epoch, but can't just subtract 1 because < 0 violates invariants.
    train_epoch = max(0, elapsed_epochs - 1)
    # nbatches = math.ceil(len(testset) / self.args.batch)
    
    class_correct = [0.0] * self.dataset.nclasses
    class_total   = [0.0] * self.dataset.nclasses
    nbatches = 0
    with torch.no_grad():
      # learner.start_eval( train_epoch, self.args.seed )
      learner.start_train( train_epoch, self.args.seed )
      # for (batch_idx, data) in enumerate(testloader):
      env.problem.reset()
      while True:
        batch_idx = nbatches
        images, labels = env.problem._current
        # images, labels = data
        if self.args.gpu:
          images = images.cuda()
          labels = labels.cuda()
        log.debug( "eval.images.shape: %s", images.shape )
        yhat = learner.forward( batch_idx, images, labels )
        log.debug( "eval.yhat: %s", yhat )
        learner.measure( batch_idx, images, labels, yhat.data )
        _, predicted = torch.max( yhat.data, 1 )
        log.debug( "eval.labels: %s", labels )
        log.debug( "eval.predicted: %s", predicted )
        c = (predicted == labels).cpu().numpy()
        log.debug( "eval.correct: %s", c )
        for i in range(len(c)):
          label = labels[i]
          class_correct[label] += c[i]
          class_total[label] += 1
        nbatches += 1
        if self.args.max_batches is not None and nbatches > self.args.max_batches:
          log.info( "==== Max batches (%s)", nbatches )
          break
        if self.args.quick_check:
          break
        env.problem._advance()
      learner.finish_eval( train_epoch )
    for i in range(self.dataset.nclasses):
      if class_total[i] > 0:
        log.info( "test %s '%s' : %s", elapsed_epochs, self.dataset.class_names[i], 
          class_correct[i] / class_total[i] )
      else:
        log.info( "test %s '%s' : None", elapsed_epochs, self.dataset.class_names[i] )
    sys.exit( 0 )
  
  def init_learner( self ):
    if self.args.agent == "ddpg":
      if self.args.representation != "blind":
        raise NotImplementedError()
      actor  = self.to_device( BlindSolarActor(
        self.observation_space, self.action_space, self.args.solar_policy_nhidden ) )
      critic = self.to_device( BlindSolarCritic(
        self.observation_space, self.action_space, self.args.solar_policy_nhidden ) )
      
      replay_memory = replay.CircularList( self.args.replay_size )
      discount = 0.99
      explore = SolarExplorePolicy(
        ddpg.ActorPolicy( actor ), self.action_space,
        GaussianWhiteNoiseProcess( sigma=0.2 ) )
      
      self.learner = ddpg.DDPGLearner(
        self.train_env, actor=actor, critic=critic, replay=replay_memory,
        discount=discount, explore=explore, batch_size=self.args.batch )
    elif self.args.agent == "double_dqn":
      if self.args.representation != "blockdrop":
        raise NotImplementedError()
      
      if self.dataset.name == "cifar10":
        defaults = resnet.defaults( self.dataset, in_channels=16 )
        stages = [resnet.ResNetStageSpec(*t)
                  for t in [(1, 16), (1, 32), (1, 64)]]
        out_channels = 64
      elif self.dataset.name == "imagenet":
        defaults = resnet.defaults( self.dataset, in_channels=64 )
        stages = [resnet.ResNetStageSpec(*t) 
                  for t in [(1,64), (1,128), (1,256), (1,512)]]
        out_channels = 512
      else:
        self.parser.error(
          "--representation={} not implemented for --dataset={}".format(
            self.args.representation, self.args.dataset ) )
      features = resnet.ResNet( resnet.ResNetBlock,
        input=defaults.input, in_shape=defaults.in_shape,
        stages=stages, output=GlobalAvgPool2d() )
      if self.args.freeze_feature_network:
        torchx.freeze( features )
        features = modules.FrozenBatchNorm( features )
        
      if self.args.env == "solar":
        qnet = ContextualQNetwork( self.observation_space, self.action_space,
                                   features, out_channels, self.args.solar_policy_nhidden )
      elif self.args.env == "cost":
        qnet = ContextualQNetwork( self.observation_space, self.action_space,
                                   features, out_channels, self.args.solar_policy_nhidden )
      elif self.args.env == "simple":
        qnet = nn.Sequential(
          features, 
          FullyConnected( out_channels, self.args.solar_policy_nhidden, bias=False ), nn.ReLU(), 
          FullyConnected( self.args.solar_policy_nhidden, self.action_space.n, bias=False ) )
      qnet.to( self.device )
      
      def make_explore():
        tokens = self.args.explore.split(",")
        if tokens[0] == "epsilon_greedy":
          spec = ",".join( tokens[1:] )
          epsilon = parameter.schedule_spec( "epsilon_greedy" )( spec )
          # self.hyperparameters.append( epsilon )
          return policy.MixturePolicy(
            policy.UniformRandomPolicy( self.train_env ),
            dqn.DqnPolicy( qnet ), epsilon )
        else:
          self.parser.error( "--explore={} incompatible with DQN" )
      explore = make_explore()
      # FIXME: What should the discount be?
      self.learner = dqn.DoubleDqnLearner(
        self.train_env, 1.0, qnet, self.make_optimizer( qnet.parameters() ),
        explore, self.args.batch, self.make_replay(),
        update_interval=self.args.update_interval,
        target_update_interval=self.args.target_update_interval,
        device=self.device )
        
      # Load or initialize parameters
      if self.args.load_checkpoint is not None:
        def load_fn( name, network ):
          load_file = self.checkpoint_mgr.get_checkpoint_file( 
            name, self.args.load_checkpoint )
          self.checkpoint_mgr.load_parameters(
            load_file, network, strict=self.args.strict_load )
        self.learner.apply_to_modules( load_fn )
      else:
        # First initialize parameters randomly because even when loading, the
        # feature network doesn't cover all of the parameters in qnet.
        self.init_network_parameters( qnet, from_file=None )
        self.init_network_parameters(
          features, from_file=self.args.load_feature_network )
      return torchx.flops( qnet, self.dataset.in_shape ).macc
    elif self.args.agent == "reinforce_bandit":
      if self.args.representation != "blockdrop":
        raise NotImplementedError()
      
      if self.dataset.name == "cifar10":
        defaults = resnet.defaults( self.dataset, in_channels=16 )
        stages = [resnet.ResNetStageSpec(*t)
                  for t in [(1, 16), (1, 32), (1, 64)]]
        out_channels = 64
      elif self.dataset.name == "imagenet":
        defaults = resnet.defaults( self.dataset, in_channels=64 )
        stages = [resnet.ResNetStageSpec(*t) 
                  for t in [(1,64), (1,128), (1,256), (1,512)]]
        out_channels = 512
      else:
        self.parser.error(
          "--representation={} not implemented for --dataset={}".format(
            self.args.representation, self.args.dataset ) )
      features = resnet.ResNet( resnet.ResNetBlock,
        input=defaults.input, in_shape=defaults.in_shape,
        stages=stages, output=GlobalAvgPool2d() )
      if self.args.freeze_feature_network:
        torchx.freeze( features )
        features = modules.FrozenBatchNorm( features )
        
      if self.args.env == "solar":
        qnet = ContextualQNetwork( self.observation_space, self.action_space,
                                   features, out_channels, self.args.solar_policy_nhidden )
      elif self.args.env == "cost":
        qnet = ContextualQNetwork( self.observation_space, self.action_space,
                                   features, out_channels, self.args.solar_policy_nhidden )
      elif self.args.env == "simple":
        qnet = nn.Sequential(
          features, 
          FullyConnected( out_channels, self.args.solar_policy_nhidden, bias=False ), nn.ReLU(), 
          FullyConnected( self.args.solar_policy_nhidden, self.action_space.n, bias=False ) )
      qnet.to( self.device )
      
      self.learner = bandit.ReinforceBanditLearner(
        self.train_env, qnet, self.make_optimizer( qnet.parameters() ),
        self.args.explore_temperature, self.args.batch, device=self.device )
      
      # Load or initialize parameters
      if self.args.load_checkpoint is not None:
        def load_fn( name, network ):
          load_file = self.checkpoint_mgr.get_checkpoint_file( 
            name, self.args.load_checkpoint )
          self.checkpoint_mgr.load_parameters(
            load_file, network, strict=self.args.strict_load )
        self.learner.apply_to_modules( load_fn )
      else:
        # First initialize parameters randomly because even when loading, the
        # feature network doesn't cover all of the parameters in qnet.
        self.init_network_parameters( qnet, from_file=None )
        self.init_network_parameters(
          features, from_file=self.args.load_feature_network )
      
      return torchx.flops( qnet, self.dataset.in_shape ).macc
    else:
      self.parser.error( "--agent={}".format(self.args.agent) )
  
  def evaluate( self, policy, nepisodes, episode_length ):
    log.info( "evaluate: nepisodes: %s; episode_length: %s",
              nepisodes, episode_length )
    Vbar = MeanAccumulator()
    Tbar = MeanAccumulator()
    for ep in range(nepisodes):
      log.info( "eval.%s.begin", ep )
      observers = [
        # rl.TrajectoryBuilder(),
        rl.EpisodeLogger( log, logging.MICRO, prefix="eval.{}.".format(ep) ),
        solar.SolarEpisodeLogger( log, logging.INFO ) ]
      (T, V) = rl.episode(
        self.eval_rng, self.eval_env, policy,
        observer=rl.EpisodeObserverList(*observers), time_limit=episode_length )
      
      # Vbar( V.squeeze()[0] )
      Vbar( V.item() )
      Tbar( T )
      
      # Format is important for log parsing
      log.info( "eval.%s.t: %s%s", ep, T, " *" if T == episode_length else "" )
      log.info( "eval.%s.v: %s", ep, V.item() )
    return Tbar.mean(), Vbar.mean()
  
  def checkpoint( self, elapsed_episodes, force_eval=False ):
    milestone = (force_eval 
                 or (elapsed_episodes % self.args.checkpoint_interval == 0))
    milestone = False
    def save_fn( name, network ):
      self.checkpoint_mgr.save_checkpoint(
        name, network, elapsed_episodes,
        data_parallel=self.args.data_parallel, persist=milestone )
    self.learner.apply_to_modules( save_fn )
    save_fn( "data_network", self.data_network )
  
    if milestone:
      # Format is important for log parsing
      log.info( "* Episode %s", elapsed_episodes )
      
      eval_policy = self.learner.eval_policy()
      
      tmean, vmean = self.evaluate( eval_policy,
        self.args.eval_episodes, self.args.eval_episode_length )
      log.info( "* eval.vmean: %s", vmean )
      log.verbose( "* eval.tmean: %s", tmean )
    if self.args.post_checkpoint_hook is not None:
      os.system( self.args.post_checkpoint_hook )
    
  def main( self ):
    # Hyperparameter updates
    # FIXME: It's still not clear exactly where these updates should happen. In
    # RL we have episode/step distinction, which is semantically different from
    # episode/batch. For now we're handling some updates within particular
    # algorithm implementations (epsilon-greedy in DQN).
    def set_epoch( epoch_idx, nbatches ):
      log.info( "training: epoch: %s; nbatches: %s", epoch_idx, nbatches )
      for hp in self.hyperparameters:
        hp.set_epoch( epoch_idx, nbatches )
        log.info( hp )
    def set_batch( batch_idx ):
      log.info( "batch: %s", batch_idx )
      for hp in self.hyperparameters:
        hp.set_batch( batch_idx )
        log.info( hp )
    
    log.info( "==================== Start ====================" )
    start = self.start_epoch
    log.info( "start: epoch: %s", start )
    moving_mean = MovingMeanAccumulator( 20 )
    # Save initial model if not resuming
    print("before checkpoint")
    if self.args.load_checkpoint is None:
      self.checkpoint( 0 )
    # Training loop
    for ep in range(start, start + self.args.train_episodes):
      print(ep)
      set_epoch( ep, nbatches=1 )
      # Update learning rate
      # FIXME: Should we do this in start_batch() for maximum Cosine?
      for param_group in self.learner.optimizer.param_groups:
        break
        param_group["lr"] = self.args.learning_rate()
      print("HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
      v = self.learner.training_episode(
        self.train_rng, ep, self.args.train_episode_length )
      #print(v)
      # if log.isEnabledFor( logging.DEBUG ):
        # bins = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        # for layer in agent.modules():
          # for p in layer.parameters():
            # h = torchx.histogram( torch.abs(p.data), bins )
            # log.debug( "train.w_mag_hist: %s\n%s", layer,
                         # torchx.pretty_histogram( h ) )
      if v is not None:
        moving_mean( v )
        log.info( "train.vmean: %s %s", ep, moving_mean.mean() )

      print("--------------------------------------------------------------")
      self.checkpoint( ep+1 )
    # Save final model if we haven't done so already
    if self.args.train_episodes % self.args.checkpoint_interval != 0:    
      self.checkpoint( self.args.train_episodes, force_eval=True )

def main():
  app = App()
  print("finished init")
  app.main()
      
if __name__ == "__main__":
  main()
