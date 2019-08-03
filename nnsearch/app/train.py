import argparse
from functools import reduce
import logging
import math
import multiprocessing
import operator
import os
import random
import shlex
import sys

import numpy.random

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.init as init
import torch.optim as optim

import nnsearch.argparse as myarg
import nnsearch.logging as mylog
import nnsearch.pytorch.act as act
from   nnsearch.pytorch.data import datasets
from   nnsearch.pytorch.dropout import BlockDropout
import nnsearch.pytorch.parameter as parameter
from   nnsearch.pytorch.models.gated import *
import nnsearch.pytorch.models.resnext as resnext
from   nnsearch.pytorch.modules import FullyConnected
from   nnsearch.statistics import MeanAccumulator

class Bunch:
  def __init__( self, **kwargs ):
    self.__dict__.update( kwargs )

# ----------------------------------------------------------------------------

class MyArgumentParser(argparse.ArgumentParser):
  def convert_arg_line_to_args( self, arg_line ):
    return shlex.split( arg_line )
    
class PrintWeights:
  def __call__( self, module ):
    if isinstance( module, (torch.nn.Conv2d, torch.nn.Linear) ):
      log.verbose( module )
      log.verbose( module.weight.data )
      
class PrintGradients:
  def __call__( self, module ):
    log.debug( module )
    if isinstance( module, (torch.nn.Conv2d, torch.nn.Linear) ):
      if module.weight.grad is not None:
        log.debug( module.weight.grad.data )
      else:
        log.debug( "None" )
        
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
      elif tokens[0] == "step":
        xs = [float(r) for r in tokens[1::2]]
        times = [int(t) for t in tokens[2::2]]
        p = parameter.StepSchedule( option_string, xs, times )
      else:
        parser.error( option_string )
      setattr(namespace, self.dest, p)
      
  return ScheduleParser

# ----------------------------------------------------------------------------

# Must have the next two lines or won't work on Windows
if __name__ == "__main__":
  multiprocessing.freeze_support()
  # On Unix, "fork" is the default start method, but it prevents us from sending process
  # logs to different files. ("spawn" is default on Windows because it doesn't have "fork")
  multiprocessing.set_start_method( "spawn" )
  
  parser = MyArgumentParser( description="RL with Equilibrium Propagation", fromfile_prefix_chars="@" )
  # TODO: Eventually this should be required=False and the default is a normal network
  type_group = parser.add_mutually_exclusive_group( required=True )
  type_group.add_argument( "--gated", action="store_true", help="Learned Bernoulli gating function." )
  type_group.add_argument( "--resnext", action="store_true", help="ResNeXt architecture." )
  type_group.add_argument( "--gated-resnext", action="store_true", help="ResNeXt architecture with path gating." )
  type_group.add_argument( "--skipnet-resnext", action="store_true", help="ResNeXt architecture with skip-net gating." )
  type_group.add_argument( "--act-resnext", action="store_true", help="Sequential ACT with ResNeXt blocks." )
  type_group.add_argument( "--block-dropout", action="store_true", help="Block-wise random dropout." )
  type_group.add_argument( "--reference-network", action="store_true", help="Use a standard reference architecture instead of anything fancy" )
  parser.add_argument( "--gating", type=str, default="bernoulli",
                       help="Create the initial network with this model of gating" )
  parser.add_argument( "--learn", type=str, default="classifier",
                       help="Which part of the model to optimize" )
  parser.add_argument( "--seed", type=int, default=None, help="Random seed" )
  parser.add_argument( "--dataset", type=str, choices=["cifar10", "fashion_mnist", "mnist"], help="Dataset" )
  parser.add_argument( "--data-directory", type=str, default="../data", help="Data root directory as expected by torchvision" )
  parser.add_argument( "--data-workers", type=int, default=0,
    help="Number of data loader worker processes. Values > 0 will make results non-reproducible." )
  parser.add_argument( "--preprocess", type=str, default=None, choices=[None, "resnet"], help="Dataset preprocessing" )
  parser.add_argument( "--input", type=str, default=".", help="Input directory" )
  parser.add_argument( "--output", type=str, default=".", help="Output directory" )
  parser.add_argument( "--log-level", type=str, choices=["NOTSET", "DEBUG", "VERBOSE", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       default="INFO", help="Logging level (we add some custom ones)" )
  parser.add_argument( "--load-checkpoint", type=int, default=None, help="Load model parameters from file" )
  parser.add_argument( "--no-strict-load", dest="strict_load", action="store_false", help="Use strict=False when loading" )
  parser.add_argument( "--checkpoint-interval", type=int, default=1, help="Epochs between saving models" )
  parser.add_argument( "--print-model", action="store_true", help="Print loaded model and exit" )
  parser.add_argument( "--evaluate", action="store_true", help="Evaluate loaded model and exit" )
  parser.add_argument( "--gpu", action="store_true", help="Use GPU" )
  # parser.add_argument( "--learning-rate", type=str, default="constant,0.01", action=parse_schedule(),
  #                      help="Schedule for learning rate; comma-separated list, first element is name of method")
  parser.add_argument( "--learning-rate", type=float, default=0.01, action=parse_schedule(),
                       help="Schedule for learning rate; comma-separated list, first element is name of method")
  parser.add_argument( "--init", type=str, choices=["kaiming", "xavier"], default="kaiming",
                       help="Weight initialization method for Conv / Linear layers." )
  parser.add_argument( "--slope-schedule", type=str, default="constant,1.0",
                       help="Slope schedule for straight-through gradient estimator;"
                            " comma-separated list, first element is name of method" )
  parser.add_argument( "--no-normalize-gated-output", dest="normalize_gated_output", action="store_false",
                       help="Do not normalize output of gated layers by number of active channels (Not recommended)" )
  parser.add_argument( "--batch", type=int, default=32, help="Batch size" )
  parser.add_argument( "--train-epochs", type=int, default=0, help="Number of training epochs" )
  parser.add_argument( "--sparsity", type=float, default=None, help="Sparsity regularization coefficient" )
  parser.add_argument( "--entropy", type=float, default=None, help="Entropy regularization coefficient" )
  parser.add_argument( "--dead-path", type=float, default=None, help="\"Dead path\" regularization coefficient" )
  parser.add_argument( "--zero-path", type=float, default=1.0,
    help="Regularization coefficient for \"at least one active path\" constraint." )
  parser.add_argument( "--no-reg-zero-path", action="store_true", help="Disable zero-path penalty" )
  parser.add_argument( "--quick-check", action="store_true", help="Do only one batch of training/testing" )
  parser.add_argument( "--max-batches", type=int, default=None, help="If set, stop training after max batches seen" )
  parser.add_argument( "--gate-conv-components", type=str, default=None,
    help="Comma-separated list of dash-separated gated conv layer specifications." )
  parser.add_argument( "--gate-fc-components", type=str, default=None,
    help="Comma-separated list of dash-separated gated FC layer specifications." )
  dropout_group = parser.add_mutually_exclusive_group()
  dropout_group.add_argument( "--dropout-rate", type=str, default=None,
    help="Proportion of blocks to de-activate when using dropout." )
  dropout_group.add_argument( "--nactive", type=str, default=None, help="Number of blocks to activate" )
  parser.add_argument( "--nactive-exponent", type=float, action=myarg.range_check(0, None), default=0,
    help="Weight loss by number of active paths." )
  parser.add_argument( "--dropout-during-eval", action="store_true", help="Use dropout during evaluation phase." )
  parser.add_argument( "--resnext-nblocks", type=int, default=3, help="Number of blocks between down-samples in ResNeXt model" )
  parser.add_argument( "--adaptive-loss-convex-p", action="store_true", help="Multiply Lacc by (1 - p) in adaptive gating loss function." )
  parser.add_argument( "--complexity-weight", action="store_true", help="Weight gating loss by complexity of gated component." )
  parser.add_argument( "--gate-control", type=str, default="uniform", help="Strategy for choosing control signal for adaptive gating." )
  parser.add_argument( "--vectorize-testing", action="store_true",
                       help="Gate by mutiplying by the gate matrix rather skipping 0-weight paths."
                            " Faster on the GPU, but loses power-saving benefits of gating." )
  parser.add_argument( "--act-ponder-factor", type=float, default=None, help="'Ponder cost' regularization coefficient." )
  parser.add_argument( "--act-bias-init", type=float, default=-3, help="Initial bias value for 'halt' modules." )
  parser.add_argument( "--act-pretrained", type=str, default=None, help="Model file to load pretrained weights from for 2-stage training." )
  args = parser.parse_args( sys.argv[1:] )
  # Parameters that vary over time
  hyperparameters = [v for v in vars(args).values() if isinstance(v, parameter.Hyperparameter)]
  
  # Logger setup
  mylog.add_log_level( "VERBOSE", logging.INFO - 5 )
  root_logger = logging.getLogger()
  root_logger.setLevel( mylog.log_level_from_string( args.log_level ) )
  # Need to set encoding or Windows will choke on ellipsis character in
  # PyTorch tensor formatting
  handler = logging.FileHandler(os.path.join( args.output, "train.log" ), "w", "utf-8")
  handler.setFormatter( logging.Formatter("%(levelname)s:%(name)s: %(message)s") )
  root_logger.addHandler(handler)
  # logging.basicConfig()
  # logging.getLogger().setLevel( logging.INFO )
  
  log = logging.getLogger( __name__ )
  log.info( "Git revision: %s", mylog.git_revision() )
  log.info( args )
  
  master_rng = random.Random( args.seed )
  def next_seed():
    seed = master_rng.randrange( 2**31 - 1 )
    random.seed( seed )
    numpy.random.seed( seed )
    torch.manual_seed( seed )
    return seed
    
  def initialize_weights( m ):
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
      if args.init == "kaiming":
        init.kaiming_normal(m.weight.data)
      elif args.init == "xavier":
        init.xavier_normal(m.weight.data)
      else:
        raise ValueError( "--init={}".format(args.init) )
      # log.debug( "init weight: %s", m.weight.data )
      if m.bias is not None:
        init.constant(m.bias.data, 0)
        # log.debug( "init bias: %s", m.bias.data )
  
  seed = next_seed()
  log.info( "==== Initializing: seed=%s", seed )
  
  for d in datasets:
    if d == args.dataset:
      dataset = datasets[d]
      break
  else:
    parser.error( "--dataset" )
  dataset.preprocess( args.preprocess )
  
  trainset    = dataset.load( root=args.data_directory, train=True )
  trainloader = torch.utils.data.DataLoader( trainset, batch_size=args.batch, shuffle=True, num_workers=args.data_workers )

  testset     = dataset.load( root=args.data_directory, train=False )
  testloader  = torch.utils.data.DataLoader( testset, batch_size=args.batch, shuffle=False, num_workers=args.data_workers )
  
  def make_dropout_cnn():
    conv_sizes = [tuple(map(int, c.split("-"))) for c in args.gate_conv_components.split(",")]
    fc_sizes   = [tuple(map(int, c.split("-"))) for c in args.gate_fc_components.split(",")]
    layers = []
    in_features = dataset.in_shape[0]
    in_shape = dataset.in_shape[1:]
    dropout_rate = float(args.dropout_rate)
    for (i, (ncomponents, component_size)) in enumerate(conv_sizes):
      layer = []
      components = [nn.Sequential(nn.Conv2d(in_features, component_size, kernel_size=3, padding=1), nn.ReLU())
                    for _ in range(ncomponents)]
      out_shapes = [(component_size, *in_shape) for _ in range(ncomponents)]
      layer.append( BlockDropout( in_features, components, out_shapes, dropout_rate ) )
      # FIXME: We should use the full-blown architecture parsing here
      if i < len(conv_sizes) - 1:
        layer.append( nn.MaxPool2d(2) )
        in_shape = tuple( s // 2 for s in in_shape )
      layers.append( nn.Sequential( *layer ) )
      in_features = ncomponents * component_size
    for s in in_shape:
      in_features *= s
    for (i, (ncomponents, component_size)) in enumerate(fc_sizes):
      if ncomponents > 1:
        components = [nn.Sequential(FullyConnected(in_features, component_size), nn.ReLU()) for _ in range(ncomponents)]
        out_shapes = [(component_size,) for _ in range(ncomponents)]
        layers.append( BlockDropout( in_features, components, out_shapes, dropout_rate ) )
      else:
        layers.append( nn.Sequential(FullyConnected(in_features, ncomponents*component_size), nn.ReLU()) )
      in_features = ncomponents * component_size
    layers.append( FullyConnected( in_features, dataset.nclasses ) )
    return nn.Sequential( *layers )
  
  def create_adaptive_dropout( method, ncomponents, path_widths ):
    nfeatures = path_widths[:]
    expansion = 4 # FIXME: assumption
    for i in range(1, len(nfeatures)):
      nfeatures[i] = expansion * path_widths[i-1]
    log.verbose( "nfeatures: %s", nfeatures )
    if method == "adaptive_independent":
      components = []
      for (i, n) in enumerate(ncomponents):
        in_features = nfeatures[i]
        log.verbose( "AdaptiveDropout.%s.in_features: %s", i, in_features )
        # TODO: Make this a parameter
        fc_sizes = [256]
        components.append( AdaptiveDropoutComponent( n, in_features, fc_sizes ) )
      return IndependentAdaptiveDropout( components )
    elif method == "adaptive_joint":
      nlayers = 3 # FIXME: assumption
      dropout_network = nn.Sequential(
        nn.Linear(1 + nfeatures[0], 256), nn.ReLU(), nn.Linear(256, nlayers) )
      return JointAdaptiveDropout( dropout_network, ncomponents )
    elif method == "adaptive_recurrent":
      # TODO: Make these parameters
      input_size = 128
      state_size = 128
      encoder_units = [256] * 3
      output_units = 256
      update_units = 256
      return RecurrentAdaptiveDropout( ncomponents, nfeatures, encoder_units,
        input_size, state_size, output_units, update_units )
    else:
      raise ValueError( "--gating={}".format(args.gating) )
  
  def dropout_fn():
    if args.dropout_rate == "uniform":
      return lambda batch_size: torch.rand(batch_size)
    else:
      dropout_rate = float(args.dropout_rate) # Trigger parse errors now
      return lambda batch_size: dropout_rate*torch.ones(batch_size)
      
  def nactive_fn( ncomponents ):
    if args.nactive == "all":
      def f( batch_size, n ):
        return n*torch.ones(batch_size)
      return f
    elif args.dropout_rate is not None:
      dropout = dropout_fn()
      def f( batch_size, n ):
        p = dropout( batch_size ).unsqueeze(-1).expand(batch_size, n)
        g = torch.bernoulli( 1 - p )
        log.debug( "nactive_fn.g: %s", g )
        nactive = torch.sum( g, dim=1 )
        log.debug( "nactive_fn.nactive: %s", nactive )
        return nactive
      return f
    elif args.nactive == "uniform":
      def f( batch_size, n ):
        p = torch.ones(batch_size, n+1)
        nactive = torch.multinomial( p, 1 ).squeeze().float()
        log.debug( "nactive_fn.nactive: %s", nactive )
        return nactive
      return f
    elif args.nactive.startswith( "uniform_annealed," ):
      tokens = args.nactive.split( "," )
      delta_t = int(tokens[1])
      inactive = parameter.LinearSchedule( "inactive", 0, 1, max(ncomponents), delta_t )
      hyperparameters.append( inactive )
      def f( batch_size, n ):
        min_active = max(n - inactive(), 0) # {0, ..., n}
        if min_active == 0:
          p = torch.ones(batch_size, n+1)
        else:
          p = torch.cat( [torch.zeros(batch_size, min_active), torch.ones(batch_size, n+1 - min_active)], dim=1 )
        nactive = torch.multinomial( p, 1 ).squeeze().float()
        log.debug( "nactive_fn.nactive: %s", nactive )
        return nactive
      return f
    elif args.nactive.startswith("linear,"):
      tokens = args.nactive.split( "," )
      end = float(tokens[1])
      def f( batch_size, n ):
        inc = end / n
        plist = [1.0 + i*inc for i in range(0, n+1)]
        p = torch.Tensor( [plist] ).expand( batch_size, n+1 )
        nactive = torch.multinomial( p, 1 ).squeeze().float()
        log.debug( "nactive_fn.nactive: %s", nactive )
        return nactive
      return f
    else:
      nactive = int(args.nactive)
      return lambda batch_size, n: min(nactive, n) * torch.ones(batch_size)
  
  # Network creation
  def create_resnext( dataset ):
    if dataset.name == "cifar10":
      nblocks = [args.resnext_nblocks] * 3
      widths = [64, 128, 256]
      ncomponents = [8, 8, 8]
      expansion = 4
      return resnext.cifar10( nblocks, widths, ncomponents, expansion )
    else:
      raise ValueError( "dataset.name = {}".format(dataset.name) )
  
  # FIXME: BernoulliGatedCnn has an idiosyncratic interface that requires other
  # approaches to be adapted to match it. Refactor.
  if args.reference_network:
    network = ReferenceNetwork( dataset )
    args.no_reg_zero_path = True
  elif args.gated:
    conv_sizes = [tuple(map(int, c.split("-"))) for c in args.gate_conv_components.split(",")]
    fc_sizes   = [tuple(map(int, c.split("-"))) for c in args.gate_fc_components.split(",")]
    network = BernoulliGatedCnn( nclasses=dataset.nclasses, in_shape=dataset.in_shape,
      conv_sizes=conv_sizes, fc_sizes=fc_sizes, normalize_output=args.normalize_gated_output )
  elif args.block_dropout:
    network = make_dropout_cnn()
    args.no_reg_zero_path = True
  elif args.gated_resnext:
    if dataset.name == "cifar10":
      network_type = Cifar10GatedResNeXt
      path_widths = [64, 128, 256]
      ncomponents = [8, 8, 8]
    elif "mnist" in dataset.name:
      network_type = MnistGatedResNeXt
      path_widths = [8, 16, 32]
      ncomponents = [8, 8, 8]
    else:
      raise ValueError( "dataset.name = {}".format(dataset.name) )
    if args.complexity_weight:
      component_weights = [
        32*32 * path_widths[0] * 4*path_widths[0],
        16*16 * 4*path_widths[0] * 4*path_widths[1],
        8*8   * 4*path_widths[1] * 4*path_widths[2]
      ]
      component_weights = Variable(torch.Tensor( [component_weights] ))
    else:
      component_weights = Variable(torch.ones( 1, 3 ))
    component_weights = fn.normalize( component_weights, p=1 )
    log.info( "component_weights: %s", component_weights )
    if args.gpu:
      component_weights = component_weights.cuda()
      
    if args.gating == "bernoulli":
      gate = BernoulliGate( dropout_fn(), ncomponents=sum( ([n]*args.resnext_nblocks for n in ncomponents), [] ),
        always_on=False, normalize=args.normalize_gated_output, dropout_during_eval=args.dropout_during_eval )
    elif args.gating == "nested":
      gate = NestedGate( nactive_fn( ncomponents ), ncomponents=sum( ([n]*args.resnext_nblocks for n in ncomponents), [] ),
        normalize=args.normalize_gated_output, dropout_during_eval=args.dropout_during_eval )
    elif args.gating.startswith( "adaptive" ):
      gate = create_adaptive_dropout( args.gating, ncomponents, path_widths )
    else:
      raise ValueError( "--gating={}".format(args.gating) )
    network = network_type( args.resnext_nblocks, path_widths=path_widths, ncomponents=ncomponents, gate=gate )
    network.vectorize_testing( args.vectorize_testing )
    args.no_reg_zero_path = True
  elif args.skipnet_resnext:
    if dataset.name == "cifar10":
      nblocks = [args.resnext_nblocks] * 3
      widths = [64, 128, 256]
      ncomponents = [8, 8, 8]
      expansion = 4
      nstages = len(nblocks)
    # elif "mnist" in dataset.name:
      # widths = [8, 16, 32]
      # ncomponents = [8, 8, 8]
    else:
      raise ValueError( "dataset.name = {}".format(dataset.name) )
      
    if args.gating == "permutation":
      gate = RandomPermutationCountGate( nactive_fn( nblocks ),
        normalize=False, dropout_during_eval=args.dropout_during_eval )
    elif args.gating == "nested":
      gate = GateList( [NestedCountGate( 
        nactive_fn( nblocks ), normalize=False, dropout_during_eval=args.dropout_during_eval )
        for _ in range(nstages)] )
    # elif args.gating.startswith( "adaptive" ):
      # gate = create_adaptive_dropout( args.gating, ncomponents, path_widths )
    else:
      raise ValueError( "--gating={}".format(args.gating) )
    network = resnext_skip_network_cifar10( gate, nblocks, widths, ncomponents, expansion )
    network.vectorize_testing( args.vectorize_testing )
    args.no_reg_zero_path = True
  elif args.act_resnext:
    if dataset.name == "cifar10":
      nblocks = [args.resnext_nblocks] * 3
      widths = [64, 128, 256]
      ncomponents = [8, 8, 8]
      expansion = 4
    # elif "mnist" in dataset.name:
      # widths = [8, 16, 32]
      # ncomponents = [8, 8, 8]
    else:
      raise ValueError( "dataset.name = {}".format(dataset.name) )
    
    network = act.resnext_sequential_cifar10( nblocks, widths, ncomponents, expansion )
    if args.act_pretrained is not None:
      pretrained = create_resnext( dataset )
      filename = args.act_pretrained
      log.info( "Loading %s", filename )
      with open( filename, "rb" ) as fin:
        # state = torch.load( fin )
        # It's not necessary to load into the network, but it's an extra sanity
        # check for whether the architecture is set up as expected.
        pretrained.load_state_dict( torch.load( fin ) )
        state = pretrained.state_dict()
      # log.debug( "pretrained state: %s", state )
      # Translate layer names; these are integer indices in the pretrained
      # network, and are remapped <i> -> F<i>
      h = torchx.hierarchical_state_dict( state )
      for i in range(1, len(h) - 1):
        stage = h[str(i)]
        new = dict( ("F{}".format(k), v) for k, v in stage.items() )
        h[str(i)] = new
      pretrained_state = torchx.flat_state_dict( h )
      # log.debug( "translated state: %s", pretrained_state )
      # Use strict=False because there are no entries for the "halt modules"
      network.load_state_dict( pretrained_state, strict=False ) 
    
    args.no_reg_zero_path = True
  elif args.resnext:
    network = create_resnext( dataset )
    args.no_reg_zero_path = True
    
    
  log.info( network )
  
  # Move to GPU
  if args.gpu:
    network.cuda()
  
  # Initialize weights
  if args.act_pretrained is not None:
    for h in network.halt_modules:
      h.apply( initialize_weights )
  else:
    network.apply( initialize_weights )
  
  def act_bias_init( m ):
    classname = m.__class__.__name__
    if "Linear" in classname:
      if m.bias is not None:
        init.constant(m.bias.data, args.act_bias_init)
        log.debug( "init act.bias: %s", m.bias.data )
  if isinstance(network, act.ACTNetwork):
    for h in network.halt_modules:
      h.apply( act_bias_init )
  
  # Regularization schemes
  def C_entropy( C ):
    eps = 1e-9
    ei = []
    for Ci in C:
      n = torch.sum( Ci )
      if n.data[0] == 0:
        ei.append( torch.zeros_like( n ) )
        break
      p = Ci / n
      log_p = torch.log( torch.clamp( p, min=eps, max=1 ) )
      ei.append( -(1.0 / math.log( 1 + Ci.numel() )) * torch.sum( p * log_p ) )
      # ei.append( -(1.0 / torch.log( n )) * torch.sum( p * log_p ) )
      # ei.append( -torch.sum( p * log_p ) )
    return sum( ei )
    
  def C_elementwise_entropy( C ):
    eps = 1e-9
    ei = []
    for Ci in C:
      p = Ci / args.batch
      log_p = torch.log( torch.clamp( p, min=eps, max=1 ) )
      log_1p = torch.log( torch.clamp( 1-p, min=eps, max=1 ) )
      ee = -(p * log_p + (1 - p) * log_1p)
      ei.append( torch.mean( ee ) )
    return sum( ei )
    
  def G_entropy( G ):
    p = G / args.batch
    log_p = torch.log( torch.max( p, 1e-5*torch.ones_like(p) ) )
    ent = -torch.sum( p * log_p, dim=1 )
    return torch.mean( ent )
    
  def G_importance( G ):
    sigma2 = torch.var( G, dim=1 )
    mu = torch.mean( G, dim=1 ) + 0.1
    cv2 = sigma2 / (mu * mu)
    return torch.sum( cv2 )
    
  def C_sparsity( Cs ):
    N = 0
    loss = 0
    for C in Cs:
      N += C.numel()
      loss += torch.sum( C )
    return 1.0 - loss / (N * args.batch)
    
  def G_zero_path( Gs ):
    loss = 0
    for G in Gs:
      Nb = G.size(0) # Batch size
      S = fn.relu( 1 - torch.sum( G, dim=1 ) ) # At least 1 should be active
      loss += torch.sum( S, dim=0 ) / Nb
    return loss
    
  def C_dead_path( Cs ):
    loss = 0
    for C in Cs:
      S = fn.relu( 1 - C ) # Number of paths that were never active
      loss += torch.sum( S )
    return loss
  
  # Hyperparameter updates
  def set_epoch( epoch_idx ):
    for hp in hyperparameters:
      hp.set_epoch( epoch_idx )
      log.info( hp )
  
  class ClassifierLearner:
    """ Learner for training the entire network. Use with a fixed gating
    strategy for two-phase gated network training.
    """
    def __init__( self ):
      self.criterion = nn.CrossEntropyLoss( reduction='none' )
      
    def start_train( self, epoch_idx, seed ):
      network.train()
      self.optimizer = optim.SGD( network.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4 )
      self.running_loss = MeanAccumulator()
      
    def start_eval( self, epoch_idx, seed ):
      network.eval()      
      
    def finish_train( self, epoch_idx ):
      log.info( "train {:d} loss: {:.3f}".format( epoch_idx, self.running_loss.mean() ) )
      
    def finish_eval( self, epoch_idx ):
      pass
      
    def measure( self, batch_idx, inputs, labels, yhat ):
      pass
      
    def loss( self, yhat, labels, *rest ):
      # Classification error
      loss = self.criterion( yhat, labels )
      criterion = torch.mean( loss )
      #log.info( "loss.criterion: %s", criterion.data[0] )
      print("CRITERION",criterion)
      try:
        print("CRITERION-TRY",criterion.item())
      except:
        print("CRITERION",criterion)
      self.running_loss( criterion.data[0] )
      
      _, predicted = torch.max( yhat.data, 1 )
      correct = torch.sum( predicted == labels.data )
      log.info( "loss.errors: %s", args.batch - correct )
      
      if args.nactive_exponent != 0:
        G = rest[0]
        cnum = torch.sum( loss )
        N = sum(network.ncomponents)
        log.verbose( "loss.Gcount: %s", G )
        Gnorm = G / N
        log.debug( "loss.Gnorm: %s", Gnorm )
        Gpow = torch.pow( Gnorm, args.nactive_exponent )
        log.debug( "loss.Gpow: %s", Gpow )
        loss *= Gpow
        cdenom = torch.sum( loss )
        c = cnum / cdenom
        loss *= c
        log.debug( "loss.weighted: %s", loss )
        loss = torch.mean( loss )
        log.debug( "loss.renorm: %s", loss )
      else:
        loss = criterion
      
      # Regularization
      if not args.no_reg_zero_path:
        G = rest[0]
        zero_path = G_zero_path( G )
        log.info( "loss.zero_path: %s", zero_path.data[0] )
        # More zero-paths is bad
        loss += args.zero_path * zero_path
      if args.sparsity is not None:
        C = rest[1]
        sparsity = C_sparsity( C )
        log.info( "loss.sparsity: %s", -sparsity.data[0] )
        # More sparsity is good
        loss -= args.sparsity * sparsity
      if args.entropy is not None:
        C = rest[1]
        entropy = C_entropy( C )
        log.info( "loss.entropy: %s", entropy.data[0] )
        # More entropy is good
        loss -= args.entropy * entropy
      if args.dead_path is not None:
        C = rest[1]
        dead_path = C_dead_path( C )
        log.info( "loss.dead_path: %s", dead_path.data[0] )
        # More dead paths is bad
        loss += args.dead_path * dead_path
      if args.act_ponder_factor is not None:
        rho_sum = torch.mean( rest[0] )
        log.info( "loss.rho_sum: %s", rho_sum.data[0] )
        loss += args.act_ponder_factor * rho_sum
      log.info( "loss.final: %s", loss.data[0] )
      return loss
    
    def forward( self, batch_idx, inputs, labels ):
      yhat, *self.rest = network( Variable(inputs) )
      return yhat
      
    def backward( self, batch_idx, yhat, labels ):
      self.optimizer.zero_grad()
      
      loss = self.loss( yhat, Variable(labels), *self.rest )
      
      # Optimization
      loss.backward()
      if log.isEnabledFor( logging.DEBUG ):
        log.debug( "Gradients" )
        network.apply( PrintGradients() )
      self.optimizer.step()
  
  def uniform_gate():
    def f( inputs, labels ):
      return Variable( torch.rand(inputs.size(0), 1).type_as(inputs) )
    return f
    
  def constant_gate( u ):
    def f( inputs, labels ):
      return Variable( (u * torch.ones(inputs.size(0), 1)).type_as(inputs) )
    return f
    
  def label_gate( c, u0, u1 ):
    def f( inputs, labels ):
      c0 = (labels != c).type_as(inputs)
      c1 = (labels == c).type_as(inputs)
      return Variable( (c0*u0 + c1*u1).unsqueeze(-1).type_as(inputs) )
    return f
  
  class AdaptiveGatingLearner:
    """ Learner for training only the "gate path" of the network.
    """
    def __init__( self, sparsity ):
      self.sparsity = sparsity
      self.criterion = nn.CrossEntropyLoss()
      
      tokens = args.gate_control.split( "," )
      if tokens[0] == "uniform":
        self.gate_control = uniform_gate()
      elif tokens[0] == "constant":
        u = float(tokens[1])
        self.gate_control = constant_gate( u )
      elif tokens[0] == "label":
        c = int(tokens[1])
        u0 = float(tokens[2])
        u1 = float(tokens[3])
        self.gate_control = label_gate( c, u0, u1 )
      else:
        raise ValueError( "--gate-control" )
      
    def train( self ):
      network.train()
      
    def eval( self ):
      network.eval()
  
    def start_epoch( self, epoch_idx, seed ):
      learning_rate = epoch_learning_rate( epoch_idx )
      slope = epoch_slope( args.slope, epoch_idx )
      log.info( "[AdaptiveGatingLearner] Epoch %s: seed=%s; learning_rate=%s; slope=%s",
                epoch_idx, seed, learning_rate, slope )
      # Don't optimize the "data" path parameters during "gate" path training
      self.optimizer = optim.SGD( network.gate.parameters(),
                                  lr=learning_rate, momentum=0.9, weight_decay=5e-4 )
      self.running_loss = MeanAccumulator()
      
    def finish_epoch( self, epoch_idx ):
      log.info( "train {:d} loss: {:.3f}".format( epoch_idx, self.running_loss.mean() ) )
      
    def loss( self, u, yhat, labels ):
      # Classification error -- don't backprop through network
      class_loss = self.criterion( yhat, labels )
      log.info( "loss.criterion: %s", torch.mean( class_loss.data ) )
      
      _, predicted = torch.max( yhat.data, 1 )
      correct = torch.sum( predicted == labels.data )
      log.info( "loss.errors: %s", args.batch - correct )
      
      ps = network.gate.ps()
      # log.debug( "ps|u|labels|predicted: %s",
        # torch.cat( [ps.data, u.data, labels.data.type_as(ps.data).unsqueeze(-1), predicted.type_as(ps.data).unsqueeze(-1)], dim=1 ) )
      Lacc = u * class_loss
      log.debug( "Lacc: %s", Lacc )
      if args.adaptive_loss_convex_p:
        Lacc *= (1.0 - ps)
      Lsparse = ps * (1.0 - u)
      log.debug( "Lsparse: %s", Lsparse )
      Lsparse *= component_weights
      log.debug( "scale(Lsparse): %s", Lsparse )
      Lsparse = torch.sum( Lsparse, dim=1 )
      log.debug( "sum(Lsparse): %s", Lsparse )
      
      L = torch.mean( Lacc + self.sparsity * Lsparse )
      self.running_loss( L.data[0] )
      return L
      
    def measure( self, batch_idx, inputs, labels, yhat ):
      _, predicted = torch.max( yhat, 1 )
      ps = network.gate.ps().detach().data
      log.verbose( "ps|u|labels|predicted: %s",
        torch.cat( [ps, self.u.data, labels.type_as(ps).unsqueeze(-1),
                    predicted.type_as(ps).unsqueeze(-1)], dim=1 ) )
    
    def forward( self, batch_idx, inputs, labels ):
      log.debug( "input.size: %s", inputs.size() )
      self.u = self.gate_control( inputs, labels )
      # self.u = Variable( torch.rand(inputs.size(0), 1).type_as(inputs) )
      network.gate.set_control( self.u )
      log.debug( "batch.u: %s", self.u )
      yhat, G, C = network( Variable(inputs) )
      return yhat
      
    def backward( self, batch_idx, yhat, labels ):
      # Zero entire network; FIXME: not necessary (wastes computation) but I
      # don't want to get burned later because I didn't do it.
      network.zero_grad()
      
      gate_loss = self.loss( self.u, yhat, Variable(labels) )
      log.info( "gate_loss: %s", gate_loss )
      
      # Optimization
      gate_loss.backward()
      if log.isEnabledFor( logging.DEBUG ):
        log.debug( "Gradients" )
        network.apply( PrintGradients() )
      self.optimizer.step()
  
  if args.learn == "classifier":
    learner = ClassifierLearner()
  elif args.learn.startswith( "gating" ):
    tokens = args.learn.split( "," )
    learn_gate = tokens[1]
    if args.gating != learn_gate:
      # We loaded a different gate method than the one we will train
      network.gate = create_adaptive_dropout( learn_gate, network.ncomponents, network.path_widths )
      if args.gpu:
        network.gate.cuda()
      log.info( "Training different network:\n%s", network )
    # Scale sparsity by CE-loss of random guessing
    sparsity = args.sparsity if args.sparsity is not None else 1.0
    learner = AdaptiveGatingLearner( sparsity*math.log(dataset.nclasses) )
  else:
    raise ValueError( "--learner={}".format( args.learner ) )
    
  # --------------------------------------------------------------------------
    
  def model_file( directory, epoch ):
    return os.path.join( directory, "model_{}.pkl".format(epoch) )
    
  def evaluate( elapsed_epochs, learner ):
    # Hyperparameters interpret their 'epoch' argument as index of the current
    # epoch; we want the same hyperparameters as in the most recent training
    # epoch, but can't just subtract 1 because < 0 violates invariants.
    train_epoch = max(0, elapsed_epochs - 1)
    set_epoch( train_epoch )
    learner.start_eval( train_epoch, seed )
    
    class_correct = [0.0] * dataset.nclasses
    class_total   = [0.0] * dataset.nclasses
    nbatches = 0
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
      c = (predicted == labels).squeeze().cpu().numpy()
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
        log.info( "test %s '%s' : %s", elapsed_epochs, dataset.class_names[i], class_correct[i] / class_total[i] )
      else:
        log.info( "test %s '%s' : None", elapsed_epochs, dataset.class_names[i] )
    
  def checkpoint( elapsed_epochs, learner ):
    with open( model_file( args.output, elapsed_epochs ), "wb" ) as fout:
      torch.save( network.state_dict(), fout )
    evaluate( elapsed_epochs, learner )
      
  if args.load_checkpoint is not None:
    filename = model_file( args.input, args.load_checkpoint )
    log.info( "Loading %s", filename )
    with open( filename, "rb" ) as fin:
      network.load_state_dict( torch.load( fin ), strict=args.strict_load )
    if args.print_model:
      for (i, m) in enumerate(network.modules()):
        # log.info( "network.%s:", m )
        for (name, param) in m.named_parameters():
          log.info( "%s: %s", name, param )
    if args.evaluate:
      evaluate( args.load_checkpoint, learner )
  
  for (name, p) in network.named_parameters():
    log.debug( "p: %s %s", name, p )
  
  if args.train_epochs <= 0:
    sys.exit( 0 )
  
  # --------------------------------------------------------------------------
  
  # Save initial model if not resuming
  seed = next_seed()
  if args.load_checkpoint is None and not args.quick_check:
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
    nbatches = 0
    set_epoch( epoch )
    learner.start_train( epoch, seed )
    for i, data in enumerate(trainloader):
      inputs, labels = data
      if args.gpu:
        inputs = inputs.cuda()
        labels = labels.cuda()
      
      yhat = learner.forward( i, inputs, labels )
      learner.backward( i, yhat, labels )
      
      nbatches += 1
      if args.max_batches is not None and nbatches >= args.max_batches:
        log.info( "==== Max batches (%s)", nbatches )
        break
      if args.quick_check:
        break
    learner.finish_train( epoch )
    if (epoch + 1) % args.checkpoint_interval == 0:
      checkpoint( epoch + 1, learner )
  # Save final model if we haven't done so already
  if args.train_epochs % args.checkpoint_interval != 0:    
    checkpoint( args.train_epochs, learner )
