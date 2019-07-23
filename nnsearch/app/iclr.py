import argparse
import ast
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
import nnsearch.pytorch.act as act
from   nnsearch.pytorch.data import datasets
from   nnsearch.pytorch.gated import blockdrop
import nnsearch.pytorch.gated.control.direct as direct
import nnsearch.pytorch.gated.densenet as densenet
import nnsearch.pytorch.gated.learner as glearner
from   nnsearch.pytorch.gated import pretrained
import nnsearch.pytorch.gated.resnext as resnext
import nnsearch.pytorch.gated.strategy as strategy
import nnsearch.pytorch.gated.vgg as vgg
from   nnsearch.pytorch.models import resnet
import nnsearch.pytorch.models.resnext as std_resnext
import nnsearch.pytorch.modules as modules
from   nnsearch.pytorch.modules import FullyConnected, GlobalAvgPool2d
import nnsearch.pytorch.parameter as parameter
import nnsearch.pytorch.torchx as torchx
from   nnsearch.statistics import MeanAccumulator

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
  parser.add_argument( "--output", type=str, default="nnsearch/nnsearchsaved/", help="Output directory" )
  load_group = parser.add_mutually_exclusive_group()
  load_group.add_argument( "--load-checkpoint", type=str, default=None,
    help="Load model parameters from file: either 'latest' or epoch number" )
  load_group.add_argument( "--load-pretrained", type=str, default=None,
    choices=["densenet169", "resnext50", "resnet50"],
    help="Load pretrained model" )
  parser.add_argument( "--convert", action="store_true",
    help="Save loaded pretrained model in nnsearch format even if not training" )
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
  aws_group = parser.add_argument_group( "aws" )
  aws_group.add_argument( "--aws-sync", type=str, default=None,
    help="If not None, a shell command to run after every epoch to sync results." )
  arch_group = parser.add_argument_group( "architecture" )
  arch_group = arch_group.add_mutually_exclusive_group( required=True )
  arch_group.add_argument( "--resnet", type=str, default=None,
    help="(Ungated) standard ResNet" )
  arch_group.add_argument( "--resnext", action="store_true",
    help="ResNeXt architecture." )
  arch_group.add_argument( "--blockdrop-resnet", type=str, default=None,
    help="Vanilla ResNet with BlockDrop gating" )
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
  resnet_group = parser.add_argument_group( "resnet" )
  resnet_group.add_argument( "--resnet-bottleneck", action="store_true",
    help="Use 'bottleneck' ResNet blocks" )
  resnext_group = parser.add_argument_group( "resnext" )
  resnext_group.add_argument( "--resnext-block", type=str, choices=["A", "C"],
    default="A", help="ResNeXt block structure" )
  resnext_group.add_argument( "--resnext-expansion", type=int, default=None,
    help="Ratio of in_channels to out_channels in each ResNeXt block. Default:"
      " dataset-specific setting from the paper." )
  resnext_group.add_argument( "--no-skip-connection-batchnorm",
    dest="skip_connection_batchnorm", action="store_false",
    help="Don't use BatchNorm on the output of skip connection downsampling"
         " layers (this is *legacy* behavior but is *incorrect* in that it"
         " does not match the ResNeXt paper." )
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
    default="bernoulli", help="Binarization method for component-wise gating." )
  gate_group.add_argument( "--control", type=str, default=None,
    help="Control method for gating." )
  gate_group.add_argument( "--gate-control", type=str, default="uniform",
    help="Strategy for choosing control signal when training adaptive gating." )
  gate_group.add_argument( "--gate-control-min",
    type=parameter.schedule_spec("--gate-control-min"), default="constant,0",
    help="Schedule for minimum value returned by --gate-control function." )
  gate_group.add_argument( "--gate-reverse-normalize", action="store_true",
    help="Scale gate vectors *up* so that their sum is equal to the number"
         " of components (default: scale sum equal 1). Necessary when using"
         " pretrained models from PyTorch. (Some say this is always the right"
         " way to do it.)" )
  gate_group.add_argument( "--no-gate-during-eval", dest="gate_during_eval",
    action="store_false", help="Disable gating during evaluation phase." )
  gate_group.add_argument( "--no-vectorize-eval", dest="vectorize_eval",
    action="store_false",
    help="Gate by skipping 0-weight paths rather than mutiplying by the gate matrix."
    " Slower on GPU, but required to realize power savings." )
  gate_group.add_argument( "--gate-temperature", type=str,
    action=parse_schedule(), default=None,
    help="Schedule for temperature parameter of gate module. A value of"
         " 'constant,0' is a hard 0-1 threshold; generally backprop doesn't"
         " work with temperature = 0 and this is used only for inference." )
  gate_group.add_argument( "--gbar-info", action="store_true",
    help="Log information about gate matrices (at the INFO level)." )
  learn_group = parser.add_argument_group( "learn" )
  learn_group.add_argument( "--learn", type=str,
    choices=["ungated", "classifier", "gate", "both"],
    default="classifier", help="Which part of the model to optimize" )
  learn_group.add_argument( "--optimizer", type=str, default="sgd",
    help="Optimizer class" )
  # learn_group.add_argument( "--learning-rate", type=str, default=None,  
  #   action=parse_schedule(), help="Learning rate schedule; comma-separated list,"
  #   " first element is name of method" )
  learn_group.add_argument( "--learning-rate", type=str, default=None, help="Learning rate schedule; comma-separated list,"
    " first element is name of method" )
  learn_group.add_argument( "--init", type=str, choices=["kaiming", "xavier"],
    default="kaiming", help="Weight initialization method for Conv / Linear layers." )
  learn_group.add_argument( "--batch", type=int, default=32, help="Batch size" )
  learn_group.add_argument( "--train-epochs", type=int, default=0,
    help="Number of training epochs" )
  learn_group.add_argument( "--max-batches", type=int, default=None,
    help="If set, stop training after max batches seen" )
  reinforce_group = parser.add_argument_group( "reinforce" )
  reinforce_group.add_argument( "--reinforce-bias-init", type=float, default=0.0,
    help="Initial value for bias in REINFORCE gating modules, to encourage"
         " using all modules at first." )
  reinforce_group.add_argument( "--reinforce-prob-range",
    type=parameter.schedule_spec("--reinforce-prob-range"),
    default="constant,0.999",
    help="If not None, restrict gate probability to a range of size <value>"
         " centered at 0.5. Values of 0 or 1 may cause numerical problems." )
  concrete_group = parser.add_argument_group( "concrete" )
  concrete_group.add_argument( "--concrete-temperature",
    type=parameter.schedule_spec("--concrete-temperature"),
    default="constant,0.1",
    help="Schedule for Concrete distribution temperature. A value of"
         " 'constant,0' recovers the deterministic limit." )
  adaptive_group = parser.add_argument_group( "adaptive" )
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
  return parser
  
# ----------------------------------------------------------------------------
# GLOBALS

parser = create_cmd_parser()
args = None
  
# ----------------------------------------------------------------------------

class GlobalAvgPool2d_Fc(nn.Module):
  def __init__( self, in_features, out_features ):
    super().__init__()
    self.fc = FullyConnected( in_features, out_features )
    
  def forward( self, x ):
    kernel_size = x.size()[2:]
    x = fn.avg_pool2d( x, kernel_size )
    return self.fc( x )

def make_gate_controller( ncomponents, always_on ):
  control_tokens = args.control.split( "," )
  control = control_tokens[0]
  if args.granularity == "count":
    n = ncomponents - 1 if always_on else ncomponents
    if control == "uniform":
      count = strategy.UniformCount( n )
    elif control == "binomial":
      p = float(control_tokens[1])
      if not (0 < p <= 1):
        parser.error("--control={}: binomial prob not in (0,1]".format(args.control))
      count = strategy.BinomialCount( n, p )
    elif control == "count":
      gcount = int(control_tokens[1])
      if not (0 <= gcount <= ncomponents):
        parser.error( "--control={}".format( args.control ) )
      count = strategy.ConstantCount( ncomponents, gcount )
      always_on = False
    if always_on:
      count = strategy.PlusOneCount( count )
    return make_gate_order( ncomponents, count )
  parser.error( "--granularity={}".format( args.granularity ) )
  
def make_gate_order( ncomponents, controller ):
  if args.order == "nested":
    return strategy.NestedCountGate(
      ncomponents, controller, gate_during_eval=args.gate_during_eval )
  elif args.order == "permutation":
    return strategy.RandomPermutationCountGate(
      ncomponents, controller, gate_during_eval=args.gate_during_eval )
  parser.error( "--order={}".format( args.order ) )
  
def make_static_gate_module( n, min_active=0 ):
  log.debug( "make_static_gate_module.n: %s", n )
  glayer = []
  if args.granularity == "count":
    glayer.append( strategy.ProportionToCount( min_active, n ) )
    glayer.append( strategy.CountToNestedGate( n ) )
    if args.order == "permutation":
      glayer.append( strategy.PermuteColumns() )
    elif args.order != "nested":
      parser.error( "--order={}".format(args.order) )
  else:
    parser.error( "--granularity={}".format(args.granularity) )
  return strategy.StaticGate( nn.Sequential( *glayer ) )

# FIXME: Generalize to different `ncomponents` in different stages
def make_gate( dataset, ncomponents, in_channels=None, always_on=False ):
  # assert( all( c == ncomponents[0] for c in ncomponents[1:] ) )
  if in_channels is not None:
    assert len(ncomponents) == len(in_channels)
  # print( in_channels )
  
  if args.granularity == "count":
    noptions = [n + 1 for n in ncomponents]
  else:
    noptions = ncomponents[:]
  
  control_tokens = args.control.split( "," )
  control = control_tokens[0]
  
  def output_layer():
    tokens = args.binarize.split( "," )
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
      alpha = parameter.Transform( args.reinforce_prob_range, transform )
      ScheduledBoundedSigmoid = modules.ScheduledParameters(
        modules.BoundedSigmoid, alpha=alpha)
      return nn.Sequential( ScheduledBoundedSigmoid( None ), strategy.ReinforceGate() )
    elif tokens[0] == "reinforce_hard":
      transform = lambda x: 0.5 + (x / 2)
      alpha = parameter.Transform( args.reinforce_prob_range, transform )
      ScheduledBoundedSigmoid = modules.ScheduledParameters(
        modules.BoundedSigmoid, alpha=alpha)
      return nn.Sequential( ScheduledBoundedSigmoid( None ), strategy.ReinforceGateHard() )
    elif tokens[0] == "reinforce_count":
      if args.granularity != "count":
        parser.error( "'--binarize=reinforce_count' requires '--granularity=count'" )
      return nn.Sequential( nn.Softmax( dim=1 ), strategy.ReinforceCountGate() )
    elif tokens[0] == "reinforce_clamp":
      f = lambda x: torch.clamp( x, min=0, max=1 )
      return nn.Sequential( modules.Lambda( f ), strategy.ReinforceGate() )
    elif tokens[0] == "concrete_bernoulli":
      temperature = args.concrete_temperature
      ScheduledConcreteBernoulli = modules.ScheduledParameters(
        strategy.ConcreteBernoulliGate, temperature=temperature )
      return nn.Sequential( nn.Sigmoid(), ScheduledConcreteBernoulli( None ) )
    elif tokens[0] == "tempered_sigmoid":
      ScheduledTemperedSigmoid = modules.ScheduledParameters(
        strategy.TemperedSigmoidGate, temperature=args.gate_temperature )
      return ScheduledTemperedSigmoid( None )
    parser.error( "Invalid combination of {control, granularity, binarize}" )
  
  if control == "lstm":
    assert( not always_on )
    assert False, "This code probably doesn't work correctly right now"
    hidden_size = int(control_tokens[1])
    input = [GlobalAvgPool2d_Fc(n, hidden_size) for n in in_channels]
    lstm = nn.LSTMCell( hidden_size+1, hidden_size )
    output = output_layer()
    return direct.RecurrentController(
      lstm, hidden_size, input=input, output=output, use_cuda=args.gpu )
  elif control == "independent":
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
      # control = nn.Sequential( FullyConnected( n+1, hidden_size ), nn.ReLU(), 
        # FullyConnected( hidden_size, c ) )
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
  elif control == "blockdrop":
    #assert not always_on
    if dataset.name == "cifar10":
      defaults = resnet.defaults( dataset, in_channels=16 )
      stages = [resnet.ResNetStageSpec(*t) for t in [(1, 16), (1, 32), (1, 64)]]
      out_channels = 64
    elif dataset.name == "imagenet":
      defaults = resnet.defaults( dataset, in_channels=64 )
      stages = [resnet.ResNetStageSpec(*t) for t in [(1,64), (1,128), (1,256), (1,512)]]
      out_channels = 512
    else:
      parser.error( "--control={} not implemented for --dataset={}".format(
                    args.control, args.dataset ) )
    features = resnet.ResNet( resnet.ResNetBlock,
      input=defaults.input, in_shape=defaults.in_shape,
      stages=stages, output=GlobalAvgPool2d() )
    total = sum( noptions )
    control = nn.Sequential(
      FullyConnected(out_channels + 1, out_channels), 
      nn.ReLU(), 
      FullyConnected(out_channels, total) )
    output = output_layer()
    controller = direct.IndependentController( features, control, output )
    return strategy.JointGate( controller, noptions )
  elif control == "static":
    # gate_modules = [make_gate_controller(c, always_on=always_on)
                    # for c in ncomponents]
    # gate = strategy.SequentialGate( gate_modules )
    min_active = 1 if always_on else 0
    ms = [make_static_gate_module( n, min_active ) for n in ncomponents]
    return strategy.SequentialGate( ms )
  elif control == "static_joint":
    min_active = 1 if always_on else 0
    gate = make_static_gate_module( sum(ncomponents) )
    return strategy.JointGate( gate, ncomponents )
  elif control == "blockdrop_nested":
    return strategy.BlockdropNestedGate( ncomponents )
  parser.error( "Invalid combination of {control, granularity, binarize}" )
  
# ----------------------------------------------------------------------------

def ungated_resnet( dataset, arch_string ):
  defaults = resnet.defaults(dataset)
  stages = ast.literal_eval(arch_string)
  stages = [resnet.ResNetStageSpec(*t) for t in stages]
  out_channels = stages[-1].nchannels
  output = nn.Sequential(
    modules.GlobalAvgPool2d(),
    modules.FullyConnected( out_channels, dataset.nclasses ) )
  network = resnet.ResNet( defaults.input, defaults.in_shape, stages, output )
  return network

def gated_densenet( dataset, arch_string ):
  nlayers = [int(t) for t in arch_string.split(",")]
  B = args.densenet_bottleneck
  C = args.densenet_compression
  k = args.densenet_growth_rate
  # Input
  defaults = densenet.defaults( dataset, k, C )
  # Rest of network
  dense_blocks = [densenet.DenseNetBlockSpec(n, B) for n in nlayers]
  gate_modules = []
  ncomponents = [spec.nlayers for spec in dense_blocks]
  in_channels = densenet.in_channels(
    k, defaults.in_shape, dense_blocks, compression=C )
  
  gate = make_gate( dataset, ncomponents, in_channels, always_on=False )
  net = densenet.GatedDenseNet( gate, k, defaults.input, defaults.in_shape,
    dataset.nclasses, dense_blocks, compression=C,
    reverse_normalize=args.gate_reverse_normalize, gbar_info=args.gbar_info )
  return net
  
def gated_resnext( dataset, arch_string ):
  # Input module
  defaults = resnext.defaults(dataset)
  expansion = (args.resnext_expansion if args.resnext_expansion is not None 
               else defaults.expansion)
  # Rest of network
  stages = eval(arch_string)
  stages = [resnext.ResNeXtStage(*t, expansion) for t in stages]
  
  ncomponents = [stage.ncomponents for stage in stages]
  log.debug( "gated_resnext.ncomponents: %s", ncomponents )
  in_channels = [defaults.in_shape[0]] + [stage.nchannels*expansion for stage in stages[:-1]]
  gate = make_gate( dataset, ncomponents, in_channels, always_on=False )
  chunk_sizes = [stage.nlayers for stage in stages]
  gate = strategy.GateChunker( gate, chunk_sizes )
  
  resnext_block_t = (resnext.GatedResNeXtBlock if args.resnext_block == "A"
                     else resnext.GatedGroupedResNeXtBlock)
  net = resnext.GatedResNeXt(
    gate, defaults.input, defaults.in_shape, dataset.nclasses, stages,
    resnext_block_t=resnext_block_t,
    skip_connection_batchnorm=args.skip_connection_batchnorm,
    reverse_normalize=args.gate_reverse_normalize )
  return net

def gated_vgg( dataset, arch_string ):
  from nnsearch.pytorch.gated.vgg import VggA, VggB, VggD, VggE
  stages = eval(arch_string)
  stages = [vgg.GatedVggStage(*t) for t in stages]
  
  ncomponents = [stage.ncomponents for stage in stages]
  in_channels = [dataset.in_shape[0]] + [stage.nchannels for stage in stages[:-1]]
  gate = make_gate( dataset, ncomponents, in_channels, always_on=args.vgg_always_on )
  chunk_sizes = [stage.nlayers for stage in stages]
  gate = strategy.GateChunker( gate, chunk_sizes )
  
  conv_stages, fc_stage = stages[:-1], stages[-1]
  net = vgg.GatedVgg( gate, dataset.in_shape, dataset.nclasses, conv_stages,
    fc_stage, batchnorm=args.vgg_batchnorm, dropout=float(args.vgg_dropout),
    reverse_normalize=args.gate_reverse_normalize )
  return net

def blockdrop_resnet( block_t, dataset, arch_string ):
  defaults = resnet.defaults(dataset)
  stages = ast.literal_eval(arch_string)
  stages = [resnet.ResNetStageSpec(*t) for t in stages]
  # Every block is its own 1-component GatedModule
  # ncomponents = [1] * sum(spec.nblocks)
  ncomponents = [stage.nblocks for stage in stages]
  in_channels = [defaults.in_shape[0]] + [stage.nchannels for stage in stages[:-1]]
  gate = make_gate( dataset, ncomponents, in_channels, always_on=False )
  out_channels = stages[-1].nchannels
  if args.resnet_bottleneck:
    out_channels *= 4
  output = nn.Sequential(
    modules.GlobalAvgPool2d(),
    modules.FullyConnected( out_channels, dataset.nclasses ) )
  net = blockdrop.BlockDropResNet(
    gate, block_t,
    input=defaults.input, in_shape=defaults.in_shape,
    stages=stages, output=output )
  return net
  
# def blockdrop_resnet( dataset, arch_string ):
  # defaults = resnet.defaults(dataset)
  # stages = ast.literal_eval(arch_string)
  # stages = [resnet.ResNetStageSpec(*t) for t in stages]
  # # Every block is its own 1-component GatedModule
  # # ncomponents = [1] * sum(spec.nblocks)
  # ncomponents = [stage.nblocks for stage in stages]
  # in_channels = [defaults.in_shape[0]] + [stage.nchannels for stage in stages[:-1]]
  # gate = make_gate( dataset, ncomponents, in_channels, always_on=False )
  # out_channels = stages[-1].nchannels
  # output = nn.Sequential(
    # modules.GlobalAvgPool2d(),
    # modules.FullyConnected( out_channels, dataset.nclasses ) )
  # net = blockdrop.BlockDropResNet(
    # gate, blockdrop.BlockDropResNetBlock,
    # input=defaults.input, in_shape=defaults.in_shape,
    # stages=stages, output=output )
  # return net
  
# def blockdrop_resnet_bottleneck( dataset, arch_string ):
  # defaults = resnet.defaults(dataset)
  
  # expansion = (args.resnext_expansion if args.resnext_expansion is not None 
               # else defaults.expansion)
  # # Rest of network
  # stages = ast.literal_eval(arch_string)
  # # FIXME: We're using the ResNeXtStage type from gated resnext for convenience
  # stages = [resnext.ResNeXtStage(*t, 1, expansion) for t in stages]
  
  # # Every block is its own 1-component GatedModule
  # # ncomponents = [1] * sum(spec.nblocks)
  # ncomponents = [stage.nblocks for stage in stages]
  # in_channels = [defaults.in_shape[0]] + [stage.nchannels for stage in stages[:-1]]
  # gate = make_gate( dataset, ncomponents, in_channels, always_on=False )
  # out_channels = stages[-1].nchannels
  # output = nn.Sequential(
    # modules.GlobalAvgPool2d(),
    # modules.FullyConnected( out_channels, dataset.nclasses ) )
  # net = blockdrop.BlockDropResNet(
    # gate, input=defaults.input, in_shape=defaults.in_shape,
    # stages=stages, output=output )
  # return net
  
# ----------------------------------------------------------------------------

def make_optimizer( parameters ):
  # PyTorch 1.0 requires default lr. Dummy value 9999 will get overwritten
  # before use.
  if args.optimizer == "sgd":
    return optim.SGD( parameters, lr=9999, momentum=0.9, weight_decay=5e-4 )
  elif args.optimizer == "adam":
    return optim.Adam( parameters, lr=9999 )
  parser.error( "--optimizer={}".format(args.optimizer) )

# ----------------------------------------------------------------------------

def uniform_gate():
  def f( inputs, labels ):
    # return Variable( torch.rand(inputs.size(0), 1).type_as(inputs) )
    umin = args.gate_control_min()
    r = 1.0 - umin
    return Variable( umin + r*torch.rand(inputs.size(0)).type_as(inputs) )
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
  
def gate_control():
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

# TODO: There's no point making the user specify the architecture if it has to
# be the same as the pretrained architecture anyway
def state_dict_from_pretrained( network, dataset ):
  # All pretrained models are specific to Imagenet
  assert args.dataset == "imagenet"
  if args.load_pretrained == "densenet169":
    return pretrained.densenet169( args, dataset )
  elif args.load_pretrained == "resnext50" and args.resnext_block == "A":
    return pretrained.resnext50a( args, dataset )
  elif args.load_pretrained == "resnext50" and args.resnext_block == "C":
    return pretrained.resnext50c( args, dataset )
  elif args.load_pretrained == "resnet50":
    return pretrained.resnet50( network, args, dataset )
  elif args.load_pretrained == "resnet101":
    return pretrained.resnet101( args, dataset )
  else:
    parser.error( "--load-pretrained={}".format(args.load_pretrained) )

def initialize_weights( m ):
  if torchx.is_weight_layer( m.__class__ ):
    log.debug( "init weights: %s", m )
    if args.init == "kaiming":
      init.kaiming_normal(m.weight.data)
    elif args.init == "xavier":
      init.xavier_normal(m.weight.data)
    else:
      parser.error( "--init={}".format(args.init) )
    if hasattr(m, "bias") and m.bias is not None:
      init.constant(m.bias.data, 0)
      
def act_bias_init( m ):
  if isinstance(m, nn.Linear):
    if m.bias is not None:
      # init.constant(m.bias.data, args.act_bias_init)
      init.normal(m.bias.data, args.act_bias_init, 0.1)
      log.debug( "init act.bias: %s", m.bias.data )
      
def reinforce_bias_init( m ):
  if isinstance(m, nn.Linear):
    if m.bias is not None:
      init.constant(m.bias.data, args.reinforce_bias_init)
      log.debug( "init reinforce.bias: %s", m.bias.data )

# ----------------------------------------------------------------------------

# Must have the next two lines or won't work on Windows
def main():
  global args, parser
  
  multiprocessing.freeze_support()
  # On Unix, "fork" is the default start method, but it prevents us from sending
  # process logs to different files. ("spawn" is default on Windows because it
  # doesn't have "fork")
  # [20190107:jhostetler] This is breaking our AWS deployment, possibly
  # because our Docker container has scikit in it; see:
  # https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
  #multiprocessing.set_start_method( "spawn" )
  
  # Parse command line
  args = parser.parse_args( sys.argv[1:] )
  
  # Parameters that vary over time
  hyperparameters = [v for v in vars(args).values() 
                     if isinstance(v, parameter.Hyperparameter)]
  
  # Logger setup
  # FIXME: Adding log levels should happen at package-level
  mylog.add_log_level( "MICRO",   logging.DEBUG - 5 )
  mylog.add_log_level( "NANO",    logging.DEBUG - 6 )
  mylog.add_log_level( "PICO",    logging.DEBUG - 7 )
  mylog.add_log_level( "FEMTO",   logging.DEBUG - 8 )
  mylog.add_log_level( "VERBOSE", logging.INFO - 5 )
  root_logger = logging.getLogger()
  root_logger.setLevel( mylog.log_level_from_string( args.log_level ) )
  # Need to set encoding or Windows will choke on ellipsis character in
  # PyTorch tensor formatting
  handler = logging.FileHandler(os.path.join( args.output, "iclr.log" ), "w", "utf-8")
  handler.setFormatter( logging.Formatter("%(levelname)s:%(name)s: %(message)s") )
  root_logger.addHandler(handler)
  # logging.basicConfig()
  # logging.getLogger().setLevel( logging.INFO )
  
  log.info( "Git revision: %s", mylog.git_revision() )
  log.info( args )
  
  master_rng = random.Random( args.seed )
  def next_seed():
    seed = master_rng.randrange( 2**31 - 1 )
    random.seed( seed )
    numpy.random.seed( seed )
    torch.manual_seed( seed )
    return seed  
  seed = next_seed()
  log.info( "==== Initializing: seed=%s", seed )
  
  try:
    dataset = datasets[args.dataset]
  except KeyError:
    parser.error( "--dataset" )
  # FIXME: ImageNet has a different preprocess method signature, and this only
  # works accidentally because it ignores its first argument. DON'T SPECIFY
  # PROPROCESSING FOR IMAGENET!!!
  dataset.preprocess( args.preprocess )

  # --------------------------------------------------------------------------
  
  network = None
  if args.resnet is not None:
    network = ungated_resnet( dataset, args.resnet )
    # Need aliases if DataParallel changes the name
    data_network = network
    gate_network = None
  else:
    if args.gated_densenet is not None:
      network = gated_densenet( dataset, args.gated_densenet )
    elif args.gated_resnext is not None:
      network = gated_resnext( dataset, args.gated_resnext )
    elif args.gated_vgg is not None:
      network = gated_vgg( dataset, args.gated_vgg )
    elif args.blockdrop_resnet is not None:
      block_t = (blockdrop.BlockDropResNetBottleneckBlock
                 if args.resnet_bottleneck
                 else blockdrop.BlockDropResNetBlock)
      network = blockdrop_resnet( block_t, dataset, args.blockdrop_resnet )
    # Need aliases if DataParallel changes the name
    data_network = network.fn
    gate_network = network.gate
  if network is None:
    parser.error( "Unsupported architecture" )
    
  log.info( network )
  def nparams( network ):
    n = 0
    nweight = 0
    for m in network.modules():
      if len(list(m.children())) > 0:
        # Parent layer
        continue
      if torchx.is_weight_layer( m.__class__ ):
        log.debug( "network.weight_layer: %s", m )
      else:
        log.debug( "other layer: %s", m )
      for p in m.parameters():
        log.debug( "parameter.size: %s", p.size() )
        n += torchx.flat_size( p.data.size() )
        if torchx.is_weight_layer( m.__class__ ):
          nweight += torchx.flat_size( p.data.size() )
    return n, nweight
  log.info( "network.nparams: %s, nweight: %s", *nparams( network ) )

  if log.isEnabledFor( logging.DEBUG ):
    for k, v in network.state_dict().items():
      log.debug( "%s: %s", k, v.shape )
  
  # --------------------------------------------------------------------------
  # Loading
  
  def model_file( directory, epoch, suffix="" ):
    filename = "model_{}.pkl{}".format( epoch, suffix )
    return os.path.join( directory, filename )
    
  def latest_checkpoints( directory ):
    return glob.glob( os.path.join( directory, "model_*.pkl.latest" ) )
    
  def epoch_of_model_file( path ):
    m = re.match( "model_([0-9]+)\.pkl\..*", os.path.basename(path) ).group(1)
    return int(m)
  
  def load_model( self, state_dict, load_gate=True, strict=True ):
    def is_gate_param( k ):
      return k.startswith( "gate." )
  
    own_state = self.state_dict()
    for name, param in state_dict.items():
      if name in own_state:
        log.verbose( "Load %s", name )
        if not load_gate and is_gate_param( name ):
          log.verbose( "Skipping gate module" )
          continue
        if isinstance(param, Parameter):
          # backwards compatibility for serialized parameters
          param = param.data
        try:
          own_state[name].copy_(param)
        except Exception:
          raise RuntimeError('While copying the parameter named {}, '
                             'whose dimensions in the model are {} and '
                             'whose dimensions in the checkpoint are {}.'
                             .format(name, own_state[name].size(), param.size()))
      elif strict:
        raise KeyError('unexpected key "{}" in state_dict'
                       .format(name))
    if strict:
      missing = set(own_state.keys()) - set(state_dict.keys())
      if not load_gate:
        missing = [k for k in missing if not is_gate_param(k)]
      if len(missing) > 0:
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))
      
  if args.load_checkpoint is not None:
    if args.load_checkpoint == "latest":
      latest = latest_checkpoints( args.input )
      if len(latest) > 1:
        parser.error( "--load-checkpoint=latest: Multiple .latest model files" )
      filename = latest[0]
      if not args.continue_:
        log.warning( "--load-checkpoint=latest but --continue not specified" )
      # Other code expects args.load_checkpoint to be an epoch index
      args.load_checkpoint = epoch_of_model_file( filename )
    else:
      # Other code expects args.load_checkpoint to be an int
      args.load_checkpoint = int(args.load_checkpoint) 
      filename = model_file( args.input, args.load_checkpoint )
    log.info( "Loading %s", filename )
    with open( filename, "rb" ) as fin:
      state_dict = torch.load(fin, map_location="cpu" if not args.gpu else None)
      load_model( network, state_dict,
                  load_gate=args.load_gate, strict=args.strict_load )
      # network.load_state_dict( torch.load( fin ), strict=args.strict_load )
    if args.print_model:
      for (i, m) in enumerate(network.modules()):
        # log.info( "network.%s:", m )
        for (name, param) in m.named_parameters():
          log.info( "%s: %s", name, param )
  elif args.load_pretrained is not None:
    state_dict = state_dict_from_pretrained( network, dataset )
    load_model( network, state_dict,
                load_gate=args.load_gate, strict=args.strict_load )
  else:
    # Initialize weights
    if args.act_pretrained is not None:
      for h in network.halt_modules:
        h.apply( initialize_weights )
    else:
      network.apply( initialize_weights )
    if args.binarize.startswith( "act" ):
      network.gate.apply( act_bias_init )
      for m in network.gate.modules():
        log.debug( "gate.act_module: %s", m )
        for p in m.parameters():
          log.debug( "act.p: %s", p )
    if args.binarize.startswith( "reinforce" ):
      network.gate.apply( reinforce_bias_init )
      for m in network.gate.modules():
        log.debug( "gate.reinforce_module: %s", m )
        for p in m.parameters():
          log.debug( "reinforce.p: %s", p )
          
  for (name, p) in network.named_parameters():
    log.micro( "p: %s %s", name, p )
  
  # Move to GPU
  if args.gpu:
    network.cuda()
  
  # --------------------------------------------------------------------------
  # Learner setup -- must be agnostic to DataParallel from here on
  
  # Multi-gpu training
  if args.data_parallel is not None:
    if not args.gpu:
      parser.error( "--data-parallel requires --gpu" )
    network = torch.nn.DataParallel(
      network, device_ids=list(range(args.data_parallel)) )
  
  # Hyperparameter updates
  def set_epoch( epoch_idx, nbatches ):
    log.info( "epoch: %s; nbatches: %s", epoch_idx, nbatches )
    if not args.continue_ and args.load_checkpoint is not None:
      epoch_idx -= args.load_checkpoint
      log.info( "effective epoch: %s", epoch_idx )
    for hp in hyperparameters:
      hp.set_epoch( epoch_idx, nbatches )
      log.info( hp )
  def set_batch( batch_idx ):
    log.info( "batch: %s", batch_idx )
    for hp in hyperparameters:
      hp.set_batch( batch_idx )
      log.info( hp )
  
  if args.learn == "ungated":
    learner = glearner.Learner( network,
      make_optimizer(data_network.parameters()), float(args.learning_rate) )
  elif args.learn == "classifier":
    learner = glearner.GatedDataPathLearner( network,
      make_optimizer(data_network.parameters()), float(args.learning_rate),
      gate_network, gate_control() )
  elif args.learn == "gate" or args.learn == "both":
    # Calculate the complexity weights of the gated modules
    complexity_weights = []
    for (m, in_shape) in network.gated_modules:
      if args.complexity_weight == "uniform":
        complexity_weights.append( 1.0 )
      elif args.complexity_weight == "macc":
        for c in m.components:
          macc = torchx.flops( c, in_shape ).macc
          log.micro( "macc: c: %s; in_shape: %s; macc: %s", c, in_shape, macc )
        macc = sum( torchx.flops( c, in_shape ).macc for c in m.components )
        log.debug( "complexity_weight: in_shape: %s; macc: %s", in_shape, macc )
        complexity_weights.append( macc )
      elif args.complexity_weight == "nparams":
        nparams = sum( torchx.nparams( c ).nparams for c in m.components )
        complexity_weights.append( nparams )
      else:
        parser.error( "--complexity-weight={}".format(args.complexity_weight) )
    log.info( "complexity_weights: %s", complexity_weights )
    log.info( "complexity total: %s", sum(complexity_weights) )
  
    # Scale sparsity by CE-loss of random guessing
    lambda_gate = args.lambda_gate*math.log(dataset.nclasses)
    if args.binarize == "reinforce":
      learner_type = glearner.ReinforceGateLearner
    elif args.binarize == "reinforce_count":
      learner_type = glearner.ReinforceCountGateLearner
    else:
      learner_type = glearner.GatePolicyLearner
    if args.learn == "gate":
      optimizer = make_optimizer( gate_network.parameters() )
      for n, p in gate_network.named_parameters():
        log.debug( "optimizing: {}".format(n) )
    elif args.learn == "both":
      optimizer = make_optimizer( network.parameters() )
      for n, p in network.named_parameters():
        log.debug( "optimizing: {}".format(n) )
    learner = learner_type( network, optimizer,
      float(args.learning_rate), gate_network, gate_control(), gate_loss=gate_loss(),
      lambda_gate=lambda_gate, component_weights=complexity_weights )
  else:
    raise parser.error( "--learn={}".format( args.learn ) )
    
  # --------------------------------------------------------------------------
  # Evaluation
    
  def evaluate( elapsed_epochs, learner ):
    # Hyperparameters interpret their 'epoch' argument as index of the current
    # epoch; we want the same hyperparameters as in the most recent training
    # epoch, but can't just subtract 1 because < 0 violates invariants.
    train_epoch = max(0, elapsed_epochs - 1)
    nbatches = math.ceil(len(testset) / args.batch)
    set_epoch( train_epoch, nbatches )
    
    class_correct = [0.0] * dataset.nclasses
    class_total   = [0.0] * dataset.nclasses
    nbatches = 0
    with torch.no_grad():
      learner.start_eval( train_epoch, seed )
      for (batch_idx, data) in enumerate(testloader):
        images, labels = data
        if args.gpu:
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
        if args.max_batches is not None and nbatches > args.max_batches:
          log.info( "==== Max batches (%s)", nbatches )
          break
        if args.quick_check:
          break
      learner.finish_eval( train_epoch )
    for i in range(dataset.nclasses):
      if class_total[i] > 0:
        log.info( "test %s '%s' : %s", elapsed_epochs, dataset.class_names[i], 
          class_correct[i] / class_total[i] )
      else:
        log.info( "test %s '%s' : None", elapsed_epochs, dataset.class_names[i] )
  
  def save_model( elapsed_epochs, force_persist=False ):
    # Save current model to tmp name
    with open( model_file(args.output, elapsed_epochs, ".tmp"), "wb" ) as fout:
      if args.data_parallel is not None:
        # See: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/19
        torch.save( network.module.state_dict(), fout )
      else:
        torch.save( network.state_dict(), fout )
    # Remove previous ".latest" checkpoints
    for f in latest_checkpoints( args.output ):
      with contextlib.suppress(FileNotFoundError):
        os.remove( f )
    # Move tmp file to latest
    os.rename( model_file(args.output, elapsed_epochs, ".tmp"),
               model_file(args.output, elapsed_epochs, ".latest") )
    if force_persist or ( elapsed_epochs % args.checkpoint_interval == 0 ):
      shutil.copy2( model_file(args.output, elapsed_epochs, ".latest"),
                    model_file(args.output, elapsed_epochs) )
  
  def checkpoint( elapsed_epochs, learner, force_eval=False ):
    save_model( elapsed_epochs, force_persist=force_eval )
    if force_eval or ( elapsed_epochs % args.checkpoint_interval == 0 ):
      evaluate( elapsed_epochs, learner )
    if args.aws_sync is not None:
      os.system( args.aws_sync )
      
  # --------------------------------------------------------------------------
  # Execution
  
  trainset    = dataset.load( root=args.data_directory, train=True )
  trainloader = torch.utils.data.DataLoader( trainset,
    batch_size=args.batch, shuffle=True, pin_memory=True,
    num_workers=args.data_workers )

  testset     = dataset.load( root=args.data_directory, train=False )
  testloader  = torch.utils.data.DataLoader( testset,
    batch_size=args.batch, shuffle=False, pin_memory=True,
    num_workers=args.data_workers )
  
  if args.convert:
    if args.load_pretrained is not None:
      save_model( 0, force_persist=True )
    else:
      log.warning( "Nothing to convert (no pretrained model loaded)" )
  
  if args.evaluate:
    elapsed_epochs = 0 if args.load_checkpoint is None else args.load_checkpoint
    evaluate( elapsed_epochs, learner )
  
  if args.train_epochs <= 0:
    sys.exit( 0 )
  
  # --------------------------------------------------------------------------
  # Training
  
  # Save initial model if not resuming
  seed = next_seed()
  if (args.load_checkpoint is None and args.load_pretrained is None 
      and not args.quick_check):
    log.info( "==== Epoch 0: seed=%s", seed )
    checkpoint( 0, learner )
  
  log.info(   "==================== Training ====================" )
  start = args.load_checkpoint if args.load_checkpoint is not None else 0
  for i in range(start):
    next_seed() # Make sure we have the same random seed when resuming
  
  # Training loop
  for epoch in range(start, start + args.train_epochs):
    seed = next_seed()
    log.info( "==== Train: Epoch %s: seed=%s", epoch, seed )
    batch_idx = 0
    nbatches = math.ceil(len(trainset) / args.batch)
    set_epoch( epoch, nbatches )
    learner.start_train( epoch, seed )
    for i, data in enumerate(trainloader):
      set_batch( batch_idx )
      inputs, labels = data
      if args.gpu:
        inputs = inputs.cuda()
        labels = labels.cuda()
      
      yhat = learner.forward( i, inputs, labels )
      learner.backward( i, yhat, labels )
      
      batch_idx += 1
      if args.max_batches is not None and batch_idx >= args.max_batches:
        log.info( "==== Max batches (%s)", batch_idx )
        break
      if args.quick_check:
        break
    learner.finish_train( epoch )
    checkpoint( epoch + 1, learner )
  # Save final model if we haven't done so already
  if args.train_epochs % args.checkpoint_interval != 0:    
    checkpoint( start + args.train_epochs, learner, force_eval=True )

if __name__ == "__main__":
  main()
