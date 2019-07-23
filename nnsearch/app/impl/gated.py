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

import nnsearch.argparse as myarg
import nnsearch.logging as mylog
from   nnsearch.pytorch.checkpoint import CheckpointManager
from   nnsearch.pytorch.data import datasets
import nnsearch.pytorch.gated.control.direct as direct
import nnsearch.pytorch.gated.densenet as densenet
import nnsearch.pytorch.gated.learner as glearner
import nnsearch.pytorch.gated.resnext as resnext
import nnsearch.pytorch.gated.strategy as strategy
import nnsearch.pytorch.gated.vgg as vgg
import nnsearch.pytorch.modules as modules
from   nnsearch.pytorch.modules import FullyConnected, GlobalAvgPool2d
import nnsearch.pytorch.parameter as parameter
import nnsearch.pytorch.torchx as torchx
from   nnsearch.statistics import MeanAccumulator

log = logging.getLogger( __name__ )
  
# ----------------------------------------------------------------------------

def is_gate_param( name ):
  return name.startswith( "gate." )
  
def initialize_weights( args ):
  if args.init == "kaiming_normal":
    weight_init_fn = init.kaiming_normal_
  elif args.init == "kaiming_uniform":
    weight_init_fn = init.kaiming_uniform_
  elif args.init == "xavier_normal":
    weight_init_fn = init.xavier_normal_
  elif args.init == "xavier_uniform":
    weight_init_fn = init.xavier_uniform_
  else:
    parser.error( "--init={}".format(args.init) )
  def impl( m ):
    if torchx.is_weight_layer( m.__class__ ):
      log.debug( "init weights: %s", m )
      weight_init_fn(m.weight.data)
      if hasattr(m, "bias") and m.bias is not None:
        init.constant_(m.bias.data, 0)
  return impl
      
def act_bias_init( args, m ):
  if isinstance(m, nn.Linear):
    if m.bias is not None:
      # init.constant(m.bias.data, args.act_bias_init)
      init.normal(m.bias.data, args.act_bias_init, 0.1)
      log.debug( "init act.bias: %s", m.bias.data )
      
def reinforce_bias_init( args, m ):
  if isinstance(m, nn.Linear):
    if m.bias is not None:
      init.constant(m.bias.data, args.reinforce_bias_init)
      log.debug( "init reinforce.bias: %s", m.bias.data )

# ----------------------------------------------------------------------------

def uniform_gate():
  def f( inputs, labels ):
    # return Variable( torch.rand(inputs.size(0), 1).type_as(inputs) )
    return Variable( torch.rand(inputs.size(0)).type_as(inputs) )
  return f
  
def constant_gate( u ):
  def f( inputs, labels ):
    # return Variable( (u * torch.ones(inputs.size(0), 1)).type_as(inputs) )
    return Variable( (u * torch.ones(inputs.size(0))).type_as(inputs) )
  return f
  
def binomial_gate( p, n ):
  def f( inputs, labels ):
    batch_size = inputs.size(0)
    ps = p * torch.ones( batch_size, n ).type_as(inputs)
    bern = torch.bernoulli( ps )
    binom = torch.sum( bern, dim=1 )
    u = binom / n
    return Variable(u)
  return f
  
def label_gate( c, u0, u1 ):
  def f( inputs, labels ):
    c0 = (labels != c).type_as(inputs)
    c1 = (labels == c).type_as(inputs)
    # return Variable( (c0*u0 + c1*u1).unsqueeze(-1).type_as(inputs) )
    return Variable( (c0*u0 + c1*u1).type_as(inputs) )
  return f
  
def gate_control( args ):
  tokens = args.gate_control.split( "," )
  if tokens[0] == "uniform":
    return uniform_gate()
  elif tokens[0] == "constant":
    u = float(tokens[1])
    return constant_gate( u )
  elif tokens[0] == "binomial":
    p = float(tokens[1])
    n = int(tokens[2])
    return binomial_gate( p, n )
  elif tokens[0] == "label":
    c = int(tokens[1])
    u0 = float(tokens[2])
    u1 = float(tokens[3])
    return label_gate( c, u0, u1 )
  else:
    raise ValueError( "--gate-control" )

def penalty_fn( tokens ):
  if tokens[0] == "hinge":
    p = float(tokens[1])
    assert( p >= 0 )
    def Lg( G, u ):
      # If more than `u` fraction of the network was used, penalize
      h = torch.max( torch.zeros_like(G), G - u )
      if p == 1:
        return h
      elif p == 2:
        return h*h
      else:
        return torch.pow( h, p )
    return Lg
  elif tokens[0] == "distance":
    p = float(tokens[1])
    assert( p >= 0 )
    def Lg( G, u ):
      d = torch.abs(G - u)
      if p == 1:
        return d
      elif p == 2:
        return d*d
      else:
        return torch.pow( d, p )
    return Lg
  elif tokens[0] == "penalty":
    def Lg( G, u ):
      return (1 - u) * G
    return Lg
  else:
    parser.error( "--gate-loss={}".format(args.gate_loss) )
    
def gate_loss():
  tokens = args.gate_loss.split( "," )
  if tokens[0] == "act":
    return glearner.act_gate_loss( penalty_fn( tokens[1:] ) )
  else:
    return glearner.usage_gate_loss( penalty_fn( tokens ) )
    
# ----------------------------------------------------------------------------

class GatedNetworkApp:
  def __init__( self, argument_parser ):
    self.parser = argument_parser
    self.checkpoint_mgr = None
    
    self._install_arguments( self.parser )
    self.args = self.parser.parse_args( sys.argv[1:] )
  
  def _install_arguments( self, parser ):
    parser.add_argument( "--seed", type=int, default=None, help="Random seed" )
    parser.add_argument( "--dataset", type=str, 
      choices=[d for d in datasets], help="Dataset" )
    parser.add_argument( "--data-directory", type=str, default="../data",
      help="Data root directory as expected by torchvision" )
    parser.add_argument( "--data-workers", type=int, default=0,
      help="Number of data loader worker processes."
      " Values > 0 will make results non-reproducible." )
    parser.add_argument( "--preprocess", type=str, default=None,
      choices=[None, "resnet"], help="Dataset preprocessing" )
    parser.add_argument( "--input", type=str, default=".", help="Input directory" )
    parser.add_argument( "--output", type=str, default=".", help="Output directory" )
    parser.add_argument( "--load-checkpoint", type=str, default=None,
      help="Load model parameters from file: either 'latest' or epoch number" )
    parser.add_argument( "--load-map-location", type=str, default=None,
      help="map_location argument to torch.load()" )
    parser.add_argument( "--continue", dest="continue_", action="store_true",
      help="Resume in the middle of an experiment. Scheduled hyperparameters"
           " count from 0 instead of the loaded checkpoint epoch." )
    parser.add_argument( "--no-load-gate", dest="load_gate", action="store_false",
      help="Do not load gate component (use when loading a network that was"
        " trained with a different gating method than the current one)." )
    parser.add_argument( "--no-strict-load", dest="strict_load", action="store_false", 
      help="Use strict=False when loading" )
    parser.add_argument( "--checkpoint-interval", type=int, default=1,
      help="Epochs between saving models" )
    parser.add_argument( "--print-model", action="store_true",
      help="Print loaded model and exit" )
    parser.add_argument( "--evaluate", action="store_true",
      help="Evaluate loaded model and exit" )
    parser.add_argument( "--gpu", action="store_true", help="Use GPU" )
    parser.add_argument( "--data-parallel", type=int, default=None,
      help="Use nn.DataParallel for multi-GPU training (single-machine only)." )
    parser.add_argument( "--quick-check", action="store_true",
      help="Do only one batch of training/testing" )
    log_group = parser.add_argument_group( "log" )
    log_group.add_argument( "--log-level", type=str,
      choices=["NOTSET", "FEMTO", "PICO", "NANO", "MICRO", 
               "DEBUG", "VERBOSE", "INFO", "WARNING", "ERROR", "CRITICAL"],
      default="INFO", help="Logging level (we add some custom ones)" )
    arch_group = parser.add_argument_group( "architecture" )
    arch_group = arch_group.add_mutually_exclusive_group( required=True )
    arch_group.add_argument( "--resnext", action="store_true",
      help="ResNeXt architecture." )
    arch_group.add_argument( "--gated-resnext", type=str, default=None,
      help="ResNeXt architecture with gating. Argument is a string containing a"
      " Python list of tuples (nlayers, ncomponents, nchannels) for each"
      " ResNeXt stage." )
    arch_group.add_argument( "--gated-densenet", type=str, default=None,
      help="DenseNet architecture with gating. Argument is comma-separated list"
        " of integers giving number of layers in each dense block." )
    arch_group.add_argument( "--gated-vgg", type=str, default=None,
      help="VGG architecture with gating. Argument is string containing Python"
      " list of tuples (nlayers, out_channels, ncomponents) for each VGG stage."
      " The last tuple specifies the fully-connected stage; all others specify"
      " convolution stages." )
    arch_group.add_argument( "--act-resnext", action="store_true",
      help="Sequential ACT with ResNeXt blocks." )
    densenet_group = parser.add_argument_group( "densenet" )
    densenet_group.add_argument( "--densenet-bottleneck", action="store_true",
      help="Use 'bottleneck' version of DenseNet." )
    densenet_group.add_argument( "--densenet-compression", type=float, default=1,
      action=myarg.range_check(0, 1), help="Use 'compressed' DenseNet with given"
      " compression rate." )
    densenet_group.add_argument( "--densenet-growth-rate", type=int,
      help="Growth rate parameter 'k' for DenseNet." )
    densenet_group.add_argument( "--densenet-input-features", type=int,
      default=None, help="Number of features computed by input layer. Default:"
      " 2 * growth_rate." )
    resnext_group = parser.add_argument_group( "resnext" )
    resnext_group.add_argument( "--resnext-expansion", type=int, default=None,
      help="Ratio of in_channels to out_channels in each ResNeXt block. Default:"
        " dataset-specific setting from the paper." )
    vgg_group = parser.add_argument_group( "vgg" )
    vgg_group.add_argument( "--vgg-dropout", type=float, default=0.5,
      action=myarg.range_check(0, 1),
      help="Dropout rate in classifier layer for VGG (0 to disable)." )
    vgg_group.add_argument( "--vgg-batchnorm", action="store_true",
      help="Use batchnorm in VGG" )
    vgg_group.add_argument( "--vgg-no-always-on", dest="vgg_always_on",
      action="store_false", help="Allow 0 active units in layers of VGG" )
    gate_group = parser.add_argument_group( "gate" )
    gate_group.add_argument( "--granularity", type=str,
      choices=["count", "independent", "component"], 
      default=None, help="Control gated modules individually, or only their number." )
    gate_group.add_argument( "--order", type=str, choices=["permutation", "nested"], 
      default=None, help="Order to disable gated modules for count gating." )
    gate_group.add_argument( "--binarize", type=str,
      # choices=["bernoulli", "bernoulli_sign", "threshold", "threshold_sign"],
      default="bernoulli", help="Binarization method for component-wise gating." )
    gate_group.add_argument( "--control", type=str, default=None,
      help="Control method for gating." )
    gate_group.add_argument( "--no-gate-during-eval", dest="gate_during_eval",
      action="store_false", help="Disable gating during evaluation phase." )
    gate_group.add_argument( "--no-vectorize-eval", dest="vectorize_eval",
      action="store_false",
      help="Gate by skipping 0-weight paths rather than mutiplying by the gate matrix."
      " Slower on GPU, but required to realize power savings." )
    gate_group.add_argument( "--gate-temperature", type=str,
      action=parameter.parse_schedule(), default=None,
      help="Schedule for temperature parameter of gate module. A value of"
           " 'constant,0' is a hard 0-1 threshold; generally backprop doesn't"
           " work with temperature = 0 and this is used only for inference." )
    gate_group.add_argument( "--gbar-info", action="store_true",
      help="Log information about gate matrices (at the INFO level)." )
    learn_group = parser.add_argument_group( "learn" )
    learn_group.add_argument( "--learn", type=str, choices=["classifier", "gate"],
      default="classifier", help="Which part of the model to optimize" )
    learn_group.add_argument( "--optimizer", type=str, default="sgd",
      help="Optimizer class" )
    learn_group.add_argument( "--learning-rate", type=str, default="constant,0.01",  
      action=parameter.parse_schedule(), help="Learning rate schedule; comma-separated list,"
      " first element is name of method" )
    learn_group.add_argument( "--init", type=str, choices=["kaiming_normal",
      "kaiming_uniform", "xavier_normal", "xavier_uniform"],
      default="kaiming_normal", help="Weight initialization method" )
    learn_group.add_argument( "--batch", type=int, default=32, help="Batch size" )
    learn_group.add_argument( "--train-epochs", type=int, default=0,
      help="Number of training epochs" )
    learn_group.add_argument( "--max-batches", type=int, default=None,
      help="If set, stop training after max batches seen" )
    learn_group.add_argument( "--post-checkpoint-hook", type=str, default=None,
      help="If not None, a shell command to run after every checkpoint. Can be"
           " used to e.g. sync results to a remote server." )
    reinforce_group = parser.add_argument_group( "reinforce" )
    reinforce_group.add_argument( "--reinforce-bias-init", type=float, default=0.0,
      help="Initial value for bias in REINFORCE gating modules, to encourage"
           " using all modules at first." )
    reinforce_group.add_argument( "--reinforce-prob-range", type=str,
      action=parameter.parse_schedule(), default="constant,0.999",
      help="If not None, restrict gate probability to a range of size <value>"
           " centered at 0.5. Values of 0 or 1 may cause numerical problems." )
    concrete_group = parser.add_argument_group( "concrete" )
    concrete_group.add_argument( "--concrete-temperature", type=str,
      action=parameter.parse_schedule(), default="constant,0.5",
      help="Schedule for Concrete distribution temperature. A value of"
           " 'constant,0' recovers the deterministic limit." )
    adaptive_group = parser.add_argument_group( "adaptive" )
    adaptive_group.add_argument( "--gate-control", type=str, default="uniform",
      help="Strategy for choosing control signal when training adaptive gating." )
    adaptive_group.add_argument( "--gate-loss", type=str, default="penalty",
      help="Loss function to encourage sparsity" )
    adaptive_group.add_argument( "--lambda-gate", type=float,
      action=myarg.range_check(0, None), default=1.0,
      help="Regularization coefficient for module usage." )
    adaptive_group.add_argument( "--complexity-weight", type=str,
      choices=["uniform", "macc", "nparams"], default="uniform",
      help="Weight gating loss by complexity measure of gated components." )
    adaptive_group.add_argument( "--adaptive-loss-convex-p", action="store_true",
      help="Multiply Lacc by (1 - p) in adaptive gating loss function." )
    act_group = parser.add_argument_group( "act" )
    act_group.add_argument( "--act-ponder-factor", type=float, default=None,
      help="'Ponder cost' regularization coefficient." )
    act_group.add_argument( "--act-bias-init", type=float, default=-3,
      help="Initial bias value for 'halt' modules." )
    act_group.add_argument( "--act-pretrained", type=str, default=None,
      help="Model file to load pretrained weights from for 2-stage training." )
  
  def init_dataset( self, args ):
    try:
      self.dataset = datasets[args.dataset]
    except KeyError:
      parser.error( "--dataset" )
    self.dataset.preprocess( args.preprocess )
  
  def load_dataset( self, args ):
    self.trainset    = self.dataset.load( root=args.data_directory, train=True )
    self.trainloader = torch.utils.data.DataLoader( self.trainset,
      batch_size=args.batch, shuffle=True, pin_memory=True,
      num_workers=args.data_workers )

    self.testset     = self.dataset.load( root=args.data_directory, train=False )
    self.testloader  = torch.utils.data.DataLoader( self.testset,
      batch_size=args.batch, shuffle=False, pin_memory=True,
      num_workers=args.data_workers )
  
  def init_gated_network_structure( self ):
    if self.args.gated_densenet is not None:
      return self.gated_densenet( self.dataset, self.args.gated_densenet )
    elif self.args.gated_resnext is not None:
      return self.gated_resnext( self.dataset, self.args.gated_resnext )
    elif self.args.gated_vgg is not None:
      return self.gated_vgg( self.dataset, self.args.gated_vgg )
    # else:
    self.parser.error( "Unsupported architecture" )
  
  def init_gated_network_parameters( self, network, from_file=None ):
    if from_file is not None:
      skip = None if self.args.load_gate else is_gate_param
      self.checkpoint_mgr.load_parameters(
        from_file, network, strict=self.args.strict_load, skip=skip,
        map_location=self.args.load_map_location )
    else:
      # Initialize weights
      if self.args.act_pretrained is not None:
        for h in network.halt_modules:
          h.apply( initialize_weights(self.args) )
      else:
        network.apply( initialize_weights(self.args) )
      if self.args.binarize.startswith( "act" ):
        network.gate.apply( act_bias_init(self.args) )
        for m in network.gate.modules():
          log.debug( "gate.act_module: %s", m )
          for p in m.parameters():
            log.debug( "act.p: %s", p )
      if self.args.binarize.startswith( "reinforce" ):
        network.gate.apply( reinforce_bias_init(self.args) )
        for m in network.gate.modules():
          log.debug( "gate.reinforce_module: %s", m )
          for p in m.parameters():
            log.debug( "reinforce.p: %s", p )
    
  def init( self, args ):
    self.args = args
    self.start_epoch = 0
    self.checkpoint_mgr = CheckpointManager( output=args.output, input=args.input )
    # self.init_dataset( args )
    # self.init_network_structure( args )
    # self.init_parameters( args )
    
  # --------------------------------------------------------------------------
  
  def make_optimizer( self, parameters ):
    # PyTorch 1.0 requires default lr. Dummy value 9999 will get overwritten
    # before use.
    if self.args.optimizer == "sgd":
      return optim.SGD( parameters, lr=9999, momentum=0.9, weight_decay=5e-4 )
    elif self.args.optimizer == "adam":
      return optim.Adam( parameters, lr=9999 )
    parser.error( "--optimizer={}".format(self.args.optimizer) )
  
  # ----------------------------------------------------------------------------
  # Network architecture

  def gated_densenet( self, dataset, arch_string ):
    nlayers = [int(t) for t in arch_string.split(",")]
    B = self.args.densenet_bottleneck
    C = self.args.densenet_compression
    k = self.args.densenet_growth_rate
    # Input
    defaults = densenet.defaults( dataset, k, C )
    # Rest of network
    dense_blocks = [densenet.DenseNetBlockSpec(n, B) for n in nlayers]
    gate_modules = []
    ncomponents = [spec.nlayers for spec in dense_blocks]
    in_channels = densenet.in_channels(
      k, defaults.in_shape, dense_blocks, compression=C )
    
    gate = self.make_gate( ncomponents, in_channels, always_on=False )
    net = densenet.GatedDenseNet( gate, k, defaults.input, defaults.in_shape,
      dataset.nclasses, dense_blocks, compression=C, gbar_info=self.args.gbar_info )
    return net
    
  def gated_resnext( self, dataset, arch_string ):
    # Input module
    defaults = resnext.defaults(dataset)
    expansion = (args.resnext_expansion if args.resnext_expansion is not None 
                 else defaults.expansion)
    # Rest of network
    stages = eval(arch_string)
    stages = [resnext.ResNeXtStage(*t, expansion) for t in stages]
    
    ncomponents = [stage.ncomponents for stage in stages]
    in_channels = ([defaults.in_shape[0]]
                   + [stage.nchannels*expansion for stage in stages[:-1]])
    gate = self.make_gate( ncomponents, in_channels, always_on=False )
    chunk_sizes = [stage.nlayers for stage in stages]
    gate = strategy.GateChunker( gate, chunk_sizes )
    
    net = resnext.GatedResNeXt(
      gate, defaults.input, defaults.in_shape, dataset.nclasses, stages )
    return net

  def gated_vgg( self, dataset, arch_string ):
    from nnsearch.pytorch.gated.vgg import VggA, VggB, VggD, VggE
    stages = eval(arch_string)
    stages = [vgg.GatedVggStage(*t) for t in stages]
    
    ncomponents = [stage.ncomponents for stage in stages]
    in_channels = ([dataset.in_shape[0]]
                   + [stage.nchannels for stage in stages[:-1]])
    gate = self.make_gate( ncomponents, in_channels, always_on=args.vgg_always_on )
    chunk_sizes = [stage.nlayers for stage in stages]
    gate = strategy.GateChunker( gate, chunk_sizes )
    
    conv_stages, fc_stage = stages[:-1], stages[-1]
    net = vgg.GatedVgg( gate, dataset.in_shape, dataset.nclasses, conv_stages,
      fc_stage, batchnorm=args.vgg_batchnorm, dropout=float(args.vgg_dropout) )
    return net
    
  # --------------------------------------------------------------------------
  # gate module
  
  def make_static_gate_module( self, n, min_active=0 ):
    glayer = []
    if self.args.granularity == "count":
      glayer.append( strategy.ProportionToCount( min_active, n ) )
      glayer.append( strategy.CountToNestedGate( n ) )
      if self.args.order == "permutation":
        glayer.append( strategy.PermuteColumns() )
      elif self.args.order != "nested":
        parser.error( "--order={}".format(self.args.order) )
    else:
      parser.error( "--granularity={}".format(self.args.granularity) )
    return strategy.StaticGate( nn.Sequential( *glayer ) )

  # FIXME: Generalize to different `ncomponents` in different stages
  def make_gate( self, ncomponents, in_channels, always_on=False ):
    # assert( all( c == ncomponents[0] for c in ncomponents[1:] ) )
    assert( len(ncomponents) == len(in_channels) )
    # print( in_channels )
    parser = self.parser
    
    if self.args.granularity == "count":
      noptions = [n + 1 for n in ncomponents]
    else:
      noptions = ncomponents[:]
    
    control_tokens = self.args.control.split( "," )
    control = control_tokens[0]
    
    def output_layer():
      tokens = self.args.binarize.split( "," )
      if tokens[0] == "bernoulli":
        clip_grad = float(tokens[1]) if len(tokens) > 1 else None
        return nn.Sequential(nn.Sigmoid(), strategy.BernoulliGate( clip_grad ))
      elif tokens[0] == "bernoulli_sign":
        return nn.Sequential( nn.Sigmoid(), strategy.BernoulliSignGate() )
      elif tokens[0] == "threshold":
        clip_grad = float(tokens[1]) if len(tokens) > 1 else None
        return nn.Sequential(nn.Tanh(), strategy.ThresholdGate( clip_grad ))
      elif tokens[0] == "threshold_sign":
        return nn.Sequential( nn.Tanh(), strategy.ThresholdSignGate() )
      elif tokens[0] == "act_one_shot":
        return nn.Sequential( nn.Sigmoid(), strategy.ActOneShotGate() )
      elif tokens[0] == "reinforce":
        transform = lambda x: 0.5 + (x / 2)
        alpha = parameter.Transform( self.args.reinforce_prob_range, transform )
        ScheduledBoundedSigmoid = modules.ScheduledParameters(
          modules.BoundedSigmoid, alpha=alpha)
        return nn.Sequential( ScheduledBoundedSigmoid( None ), strategy.ReinforceGate() )
      elif tokens[0] == "reinforce_count":
        if self.args.granularity != "count":
          parser.error( "'--binarize=reinforce_count' requires '--granularity=count'" )
        return nn.Sequential( nn.Softmax( dim=1 ), strategy.ReinforceCountGate() )
      elif tokens[0] == "reinforce_clamp":
        f = lambda x: torch.clamp( x, min=0, max=1 )
        return nn.Sequential( modules.Lambda( f ), strategy.ReinforceGate() )
      elif tokens[0] == "concrete_bernoulli":
        temperature = self.args.concrete_temperature
        ScheduledConcreteBernoulli = modules.ScheduledParameters(
          strategy.ConcreteBernoulliGate, temperature=temperature )
        return nn.Sequential( nn.Sigmoid(), ScheduledConcreteBernoulli( None ) )
      elif tokens[0] == "tempered_sigmoid":
        ScheduledTemperedSigmoid = modules.ScheduledParameters(
          strategy.TemperedSigmoidGate, temperature=self.args.gate_temperature )
        return ScheduledTemperedSigmoid( None )
      parser.error( "Invalid combination of {control, granularity, binarize}" )
    
    if control == "independent":
      assert( not always_on )
      hidden_sizes = [int(n) for n in control_tokens[1:]]
      def control_modules( n, c ):
        input = GlobalAvgPool2d()
        in_size = n + 1 # +1 for the control signal
        out_sizes = hidden_sizes + [c]
        layers = []
        for (i, out_size) in enumerate(out_sizes):
          if i > 0:
            layers.append( nn.ReLU() )
          layers.append( FullyConnected(in_size, out_size) )
          in_size = out_size
        control = nn.Sequential( *layers )
        return (input, control)
      ms = []
      for (n, c) in zip(in_channels, noptions):
        input, control = control_modules( n, c )
        output = output_layer()
        ms.append( direct.IndependentController( input, control, output ) )
      return strategy.SequentialGate( ms )
    elif control == "blind":
      assert( not always_on )
      hidden_size = int(control_tokens[1])
      def control_module( n ):
        return nn.Sequential( FullyConnected(1, hidden_size), nn.ReLU(),
          FullyConnected(hidden_size, n) )
      ms = [direct.BlindController(control_module(n), output_layer())
            for n in noptions]
      return strategy.SequentialGate( ms )
    elif control == "static":
      min_active = 1 if always_on else 0
      ms = [self.make_static_gate_module( n, min_active ) for n in ncomponents]
      return strategy.SequentialGate( ms )
    parser.error( "Invalid combination of {control, granularity, binarize}" )

  