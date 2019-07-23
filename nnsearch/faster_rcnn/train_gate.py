import argparse
from   functools import reduce
import logging
import math
import multiprocessing
import operator
import os
import random
import shlex
import sys

import matplotlib
from   tqdm import tqdm

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
from   torch.utils import data as torchdata

import nnsearch.argparse as myarg
from   nnsearch.faster_rcnn.data.dataset import (
  Dataset, TestDataset, inverse_normalize)
from   nnsearch.faster_rcnn.model import FasterRCNNVGG16
from   nnsearch.faster_rcnn.model.gated_densenet import FasterRCNNDenseNet169BC
from   nnsearch.faster_rcnn.trainer import FasterRCNNTrainer
from   nnsearch.faster_rcnn.utils import array_tool as at
from   nnsearch.faster_rcnn.utils.vis_tool import visdom_bbox
from   nnsearch.faster_rcnn.utils.eval_tool import eval_detection_voc
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

#import nnsearch
import nnsearch.logging as mylog

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

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

class Main:
  def __init__( self, args ):
    self.args = args
    self.checkpoint_mgr = CheckpointManager( output=args.output, input=args.input )
    self.gate_control = gate_control( args )
    
    # Parameters that vary over time
    self.hyperparameters = [v for v in vars(args).values() 
                            if isinstance(v, parameter.Hyperparameter)]
    
    if args.feature_network == "gated_densenet169bc":
      network = gated_densenet169bc()
    else:
      parser.error( "--feature-network={}".format(args.feature_network) )
    log.info( network )
    log.info( "network.nparams: %s, nweight: %s", *nparams( network ) )
    
    # Load the feature network weights
    if args.load_feature_network is not None:
      self.checkpoint_mgr.load_parameters( args.load_feature_network, network )
    elif args.load_checkpoint is not None:
      # TODO: Implement checkpoint loading for RCNN model
      pass
    else:
      raise RuntimeError( "no feature network weights to load" )
    
    self.faster_rcnn = FasterRCNNDenseNet169BC( network )
    self.trainer = FasterRCNNTrainer(self.faster_rcnn).cuda()

    log.info('Loading data...')
    # Note: Implementation only supports batch_size=1
    self.trainset = Dataset(self.args)
    self.trainloader = torchdata.DataLoader(
      self.trainset, batch_size=1, shuffle=True, pin_memory=True,
      num_workers=self.args.data_workers )
    self.testset = TestDataset(self.args)
    self.testloader = torchdata.DataLoader(
      self.testset, batch_size=1, shuffle=False, pin_memory=True,
      num_workers=self.args.data_workers )
    log.debug('...loading complete')
    
  def run( self ):
    # TODO: Implement evaluate-only, etc.
    self.train()
                            
  # Hyperparameter updates
  def set_epoch( self, epoch_idx, nbatches ):
    log.info( "epoch: %s; nbatches: %s", epoch_idx, nbatches )
    if not self.args.continue_ and self.args.load_checkpoint is not None:
      epoch_idx -= self.args.load_checkpoint
      log.info( "effective epoch: %s", epoch_idx )
    for hp in self.hyperparameters:
      hp.set_epoch( epoch_idx, nbatches )
      log.info( hp )
      
  def set_batch( self, batch_idx ):
    log.verbose( "batch: %s", batch_idx )
    for hp in self.hyperparameters:
      hp.set_batch( batch_idx )
      log.verbose( hp )

  def evaluate(self, dataloader):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
      sizes = [sizes[0][0], sizes[1][0]]
      
      # Gating modifications
      # FIXME: Normally we could infer cuda from the type of imgs, but for some
      # reason those get cuda-fied in the call to predict() instead of outside
      # like a normal person would do it (it actually happens in 
      # .util.array_tool.totensor() which moves to cuda() *by default*?!)
      u = self.gate_control( imgs, None ).cuda()
      self.faster_rcnn.set_gate_control( u )
      
      pred_bboxes_, pred_labels_, pred_scores_ = self.faster_rcnn.predict(imgs, [sizes])
      gt_bboxes += list(gt_bboxes_.numpy())
      gt_labels += list(gt_labels_.numpy())
      gt_difficults += list(gt_difficults_.numpy())
      pred_bboxes += pred_bboxes_
      pred_labels += pred_labels_
      pred_scores += pred_scores_
      
      if self.args.max_batches is not None and ii == self.args.max_batches - 1:
        break

    result = eval_detection_voc(
      pred_bboxes, pred_labels, pred_scores,
      gt_bboxes, gt_labels, gt_difficults,
      use_07_metric=True)
    return result

  def _checkpoint( self, epoch ):
    log.info( "eval.%s.loss: %s", epoch, str(self.trainer.get_meter_data()))
    log.info( "eval.%s.rpn_cm: %s",
              epoch, str(self.trainer.rpn_cm.value().tolist()) )
  
    persist = epoch % self.args.checkpoint_interval == 0
    self.checkpoint_mgr.save_checkpoint(
      "rcnn", self.trainer.faster_rcnn, epoch,
      data_parallel=(self.args.data_parallel is not None), persist=persist )
    if persist:
      eval_result = self.evaluate( self.testloader )
      for k, v in eval_result.items():
        log.info( "eval.%s.%s: %s", epoch, k, v )
    
  def train(self):
    best_map = 0
    for epoch in range(self.args.train_epochs):
      # log.info( "==== Train: Epoch %s: seed=%s", epoch, seed )
      log.info( "==== Train: Epoch %s", epoch )
      batch_idx = 0
      nbatches = math.ceil(len(self.trainset) / self.args.batch)
      self.set_epoch( epoch, nbatches )
    
      self.trainer.reset_meters()
      
      # Update learning rate
      # FIXME: Should we do this in start_batch() for maximum Cosine?
      for param_group in self.trainer.optimizer.param_groups:
        param_group["lr"] = self.args.learning_rate()
      
      for ii, (img, bbox_, label_, scale) in tqdm(enumerate(self.trainloader)):
        self.set_batch( ii )
        
        scale = at.scalar(scale)
        img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
        # Has to happen after cuda() but before Variable()
        u = self.gate_control( img, None )
        self.trainer.faster_rcnn.set_gate_control( u )
        img, bbox, label = Variable(img), Variable(bbox), Variable(label)
        
        self.trainer.train_step(img, bbox, label, scale)
        
        if self.args.max_batches is not None and ii == self.args.max_batches-1:
          break
      
      elapsed_epochs = epoch + 1
      self._checkpoint( elapsed_epochs )
              
# ----------------------------------------------------------------------------

def make_static_gate_module( n, min_active=0 ):
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
def make_gate( ncomponents, in_channels, always_on=False ):
  # assert( all( c == ncomponents[0] for c in ncomponents[1:] ) )
  assert( len(ncomponents) == len(in_channels) )
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
      # alpha = 0.5 + (args.reinforce_prob_range / 2)
      ScheduledBoundedSigmoid = modules.ScheduledParameters(
        modules.BoundedSigmoid, alpha=alpha)
      # return nn.Sequential( modules.BoundedSigmoid( alpha ), strategy.ReinforceGate() )
      return nn.Sequential( ScheduledBoundedSigmoid( None ), strategy.ReinforceGate() )
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
  elif control == "static":
    # gate_modules = [make_gate_controller(c, always_on=always_on)
                    # for c in ncomponents]
    # gate = strategy.SequentialGate( gate_modules )
    min_active = 1 if always_on else 0
    ms = [make_static_gate_module( n, min_active ) for n in ncomponents]
    return strategy.SequentialGate( ms )
  parser.error( "Invalid combination of {control, granularity, binarize}" )
  
# ----------------------------------------------------------------------------

def gated_densenet( dataset, nlayers, k, B, C ):
  # Input
  defaults = densenet.defaults( dataset, k, C )
  # Rest of network
  dense_blocks = [densenet.DenseNetBlockSpec(n, B) for n in nlayers]
  gate_modules = []
  ncomponents = [spec.nlayers for spec in dense_blocks]
  in_channels = densenet.in_channels(
    k, defaults.in_shape, dense_blocks, compression=C )
  
  gate = make_gate( ncomponents, in_channels, always_on=False )
  net = densenet.GatedDenseNet( gate, k, defaults.input, defaults.in_shape,
    dataset.nclasses, dense_blocks, compression=C, gbar_info=args.gbar_info )
  return net
  
def gated_densenet169bc():
  nlayers = [6, 12, 32, 32]
  k = 32
  B = True
  C = 0.5
  return gated_densenet( datasets["imagenet"], nlayers, k, B, C )
  
def gated_resnext( dataset, arch_string ):
  # Input module
  defaults = resnext.defaults(dataset)
  expansion = (args.resnext_expansion if args.resnext_expansion is not None 
               else defaults.expansion)
  # Rest of network
  stages = eval(arch_string)
  stages = [resnext.ResNeXtStage(*t, expansion) for t in stages]
  
  ncomponents = [stage.ncomponents for stage in stages]
  in_channels = [defaults.in_shape[0]] + [stage.nchannels*expansion for stage in stages[:-1]]
  gate = make_gate( ncomponents, in_channels, always_on=False )
  chunk_sizes = [stage.nlayers for stage in stages]
  gate = strategy.GateChunker( gate, chunk_sizes )
  
  net = resnext.GatedResNeXt(
    gate, defaults.input, defaults.in_shape, dataset.nclasses, stages )
  return net

def gated_vgg( dataset, arch_string ):
  from nnsearch.pytorch.gated.vgg import VggA, VggB, VggD, VggE
  stages = eval(arch_string)
  stages = [vgg.GatedVggStage(*t) for t in stages]
  
  ncomponents = [stage.ncomponents for stage in stages]
  in_channels = [dataset.in_shape[0]] + [stage.nchannels for stage in stages[:-1]]
  gate = make_gate( ncomponents, in_channels, always_on=args.vgg_always_on )
  chunk_sizes = [stage.nlayers for stage in stages]
  gate = strategy.GateChunker( gate, chunk_sizes )
  
  conv_stages, fc_stage = stages[:-1], stages[-1]
  net = vgg.GatedVgg( gate, dataset.in_shape, dataset.nclasses, conv_stages,
    fc_stage, batchnorm=args.vgg_batchnorm, dropout=float(args.vgg_dropout) )
  return net

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
    raise ValueError( "--gate-loss" )
    
def gate_loss( args ):
  tokens = args.gate_loss.split( "," )
  if tokens[0] == "act":
    return glearner.act_gate_loss( penalty_fn( tokens[1:] ) )
  else:
    return glearner.usage_gate_loss( penalty_fn( tokens ) )
    
# ----------------------------------------------------------------------------

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

if __name__ == '__main__':
    
    parser = MyArgumentParser(description='Train FasterRCNN',
      fromfile_prefix_chars="@", allow_abbrev=False )
    parser.add_argument('--voc-data-dir', '-vdd', type=str, default='/data1/nnsearch/data/VOCdevkit/VOC2007/')
    parser.add_argument('--min-size', type=int, default=600)
    parser.add_argument('--max-size', type=int, default=1000)
    # parser.add_argument('--num-workers', type=int, default=8)
    # parser.add_argument('--test-num-workers', type=int, default=8)
    parser.add_argument('--rpn-sigma',default=3.)
    parser.add_argument('--roi-sigma', default=1.)
    parser.add_argument('--weight-decay', type=float, default=0.005)
    parser.add_argument('--lr-decay', type=float, default=0.1)
    # parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--env', type=str, default='faster-rcnn')
    parser.add_argument('--port','-p', type=int, default=8097)
    parser.add_argument('--plot-every', type=int, default=40)
    parser.add_argument('--data', '-d', type=str, default='voc')
    parser.add_argument('--pretrained-model', type=str, default='vgg', choices=['vgg', 'gated-vgg'])
    parser.add_argument('--epoch', type=int, default=14)
    parser.add_argument('--use-adam', action='store_true')
    parser.add_argument('--use-drop', action='store_true')
    parser.add_argument('--use-chainer', action='store_true')
    parser.add_argument('--debug-file', type=str, default='/tmp/debugf')
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--load-path', default=None)
    parser.add_argument('--caffe-pretrain', action='store_true')
    parser.add_argument('--caffe-pretrain-path', type=str, default='checkpoints/vgg16-caffe.pth')
    parser.add_argument('--log-level',default='VERBOSE', choices=['VERBOSE', 'DEBUG', 'INFO'])
    parser.add_argument( "--aws-sync", type=str, default=None,
                help="If not None, a shell command to run after every epoch to sync results." )
    # ------------------------------------------------------------------------
    parser.add_argument( "--input", type=str, default="." )
    parser.add_argument( "--output", type=str, default="." )
    parser.add_argument( "--data-directory", type=str, default="../data",
      help="Data root directory as expected by torchvision" )
    parser.add_argument( "--data-workers", type=int, default=0,
      help="Number of data loader worker processes."
      " Values > 0 will make results non-reproducible." )
    parser.add_argument( "--feature-network", type=str,
      choices=["gated_densenet169bc"] )
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument( "--load-checkpoint", type=str, default=None,
      help="Load model parameters from file: either 'latest' or epoch number" )
    load_group.add_argument( "--load-feature-network", type=str, default=None,
      help="Path to pre-trained feature network" )
    parser.add_argument( "--continue", dest="continue_", action="store_true",
      help="Resume in the middle of an experiment. Scheduled hyperparameters"
           " count from 0 instead of the loaded checkpoint epoch." )
    parser.add_argument( "--checkpoint-interval", type=int, default=1,
      help="Epochs between saving models" )
    parser.add_argument( "--print-model", action="store_true",
      help="Print loaded model and exit" )
    parser.add_argument( "--evaluate", action="store_true",
      help="Evaluate loaded model and exit" )
    # parser.add_argument( "--gpu", action="store_true", help="Use GPU" )
    parser.add_argument( "--data-parallel", type=int, default=None,
      help="Use nn.DataParallel for multi-GPU training (single-machine only)." )
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
      action=parse_schedule(), default=None,
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
    learn_group.add_argument( "--learning-rate", type=str, default=None,  
      action=parse_schedule(), help="Learning rate schedule; comma-separated list,"
      " first element is name of method" )
    learn_group.add_argument( "--init", type=str, choices=["kaiming", "xavier"],
      default="kaiming", help="Weight initialization method for Conv / Linear layers." )
    learn_group.add_argument( "--batch", type=int, default=1, choices=[1],
      help="Batch size (FasterRCNN only supports batch=1)" )
    learn_group.add_argument( "--train-epochs", type=int, default=0,
      help="Number of training epochs" )
    learn_group.add_argument( "--max-batches", type=int, default=None,
      help="If set, stop training after max batches seen" )
    reinforce_group = parser.add_argument_group( "reinforce" )
    reinforce_group.add_argument( "--reinforce-bias-init", type=float, default=0.0,
      help="Initial value for bias in REINFORCE gating modules, to encourage"
           " using all modules at first." )
    reinforce_group.add_argument( "--reinforce-prob-range", type=str,
      action=parse_schedule(), default="constant,0.999",
      help="If not None, restrict gate probability to a range of size <value>"
           " centered at 0.5. Values of 0 or 1 may cause numerical problems." )
    concrete_group = parser.add_argument_group( "concrete" )
    concrete_group.add_argument( "--concrete-temperature", type=str,
      action=parse_schedule(), default="constant,0.5",
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

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    
    #mylog.add_log_level( "MICRO",   logging.DEBUG - 5 )
    #mylog.add_log_level( "NANO",    logging.DEBUG - 6 )
    #mylog.add_log_level( "PICO",    logging.DEBUG - 7 )
    #mylog.add_log_level( "FEMTO",   logging.DEBUG - 8 )
    mylog.add_log_level( "VERBOSE", logging.INFO - 5 )
    root_logger = logging.getLogger()
    root_logger.setLevel( mylog.log_level_from_string( args.log_level ) )
    handler = logging.FileHandler(os.path.join( args.output, "faster_rcnn.log" ), "w", "utf-8")
    handler.setFormatter( logging.Formatter("%(levelname)s:%(name)s: %(message)s") )
    root_logger.addHandler(handler)
    # logging.basicConfig()
    # logging.getLogger().setLevel( logging.INFO )
    log = logging.getLogger( __name__ )
    log.info( "Git revision: %s", mylog.git_revision() )
    log.info(args)
    
    # --------------------------------------------------------------------------
  
    

    app = Main( args )
    app.run()

