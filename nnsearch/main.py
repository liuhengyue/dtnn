import argparse
import itertools
import json
import logging
import multiprocessing
import os
import random
import sys

import nnsearch.backend.architect
import nnsearch.bandit
from nnsearch.crossentropy import cross_entropy_search
import nnsearch.flat
import nnsearch.gpu
from nnsearch.hcsearch import hcsearch
import nnsearch.nnspace as nnspace
import nnsearch.nnspace.wistuba
import nnsearch.nnspec as nnspec
import nnsearch.objective as objective
import nnsearch.storage

import nnsearch.mcts

# ----------------------------------------------------------------------------

# FIXME: Copy-pasted from feedforward. We can't import the original code
# because it imports theano and that can't happen in the main process.
def get_parser():
    def log_level( s ):
      return getattr( logging, s.upper() )
    
    parser = argparse.ArgumentParser(description='Load and preprocess data source.', fromfile_prefix_chars='@')
    parser.add_argument('-center', action='store_true', default=False, help="Center training set to have zero mean and unit variance. Center test set using mean/var of training set.")
    parser.add_argument('-batch_balance', action='store_true', default=False, help="Balance classes across batches.")
    parser.add_argument('-batch_shuffle', action='store_true', default=False, help="Shuffle the order of instances -> batches.")
    parser.add_argument('-batch', action='store', type=int, default=32, help="Number of instances in a batch.")
    parser.add_argument('-seed', action='store', type=int, default=42, help="Random seed for shuffling/balancing of instances.")
    parser.add_argument('-level', action='store', type=log_level, default=logging.INFO, help="Log level as in python logging package.")
    parser.add_argument('-output', action='store', type=str, default=".", help="Directory to store output models and log files.")
    parser.add_argument('-x_type', action='store', default='float32', help="Datatype for input features.")
    parser.add_argument('-y_type', action='store', default='int32', help="Datatype for labels.")
    
    def valid_ratio(a_num):
        if float(a_num) >= 1. or float(a_num) <= 0.:
            raise argparse.ArgumentTypeError('ratio should be between 0 and 1 (exclusive).')
        return float(a_num)
    
    # parser = dataset.get_parser()
    parser.add_argument('dataset', action='store', help="Filename (.pkl.gz or delimited) or URL of dataset to load")
    parser.add_argument('-headings', action='store_true', default=False, help="(Unimpl) If headings are present, only for delimited files.")
    parser.add_argument('-separator', action='store', type=str, default=',', help="(Unimpl) Column separator, only for delimited files.")
    parser.add_argument('-train_ratio', action='store', type=valid_ratio, default=0.6, 
        help="Fraction of data to use for training. Remaining is split evenly between test and validation sets.")
    parser.add_argument('-balance', action='store_true', default=False, help="Balance classes across train/test/valid sets.")
    parser.add_argument('-shuffle', action='store_true', default=False, help="Shuffle instances between train/test/valid sets (using train_ratio).")
    
    # parser = simple_load_data.get_parser()
    parser.add_argument('arch', action='store', help="Network architecture.")
    parser.add_argument('-learning_rate', action='store', type=float, default=1e-3, help="Learning rate for SGD.")
    parser.add_argument('-decay', action='store', type=float, default=.5, help="Learning rate decay after each epoch.")
    parser.add_argument('-nepoch', action='store', type=int, default=1, help="Number of epochs of training.")
    parser.add_argument('-model', action='store', type=str, default=None, help="Pretrained fp32 model.")
    
    return parser

if __name__ == "__main__":
  # On Unix, "fork" is the default start method, but it prevents us from sending process
  # logs to different files. ("spawn" is default on Windows because it doesn't have "fork")
  multiprocessing.set_start_method( "spawn" )
  
  parser = get_parser()
  input_group = parser.add_mutually_exclusive_group( required=True )
  input_group.add_argument( "--json", type=str, default=None, help="File containing JSON search space specification." )
  input_group.add_argument( "--arch-list", type=str, default=None, help="File containing Architect strings, one per line." )
  nnsearch_group = parser.add_argument_group( "nnsearch" )
  nnsearch_group.add_argument( "--seed", type=int, default=None, help="Random seed" )
  nnsearch_group.add_argument( "--nworkers", type=int, default=1, help="Number of worker processes." )
  nnsearch_group.add_argument( "--gpu-workers", type=str, default=None, help="Comma-separated list of GPU numbers (c.f. CUDA_VISIBLE_DEVICES)"
    " to use for worker processes. If shorter than --nworkers, remaining workers will use CPUs only." )
  nnsearch_group.add_argument( "--gpu-names", type=str, default=None, help="Comma-separated GPU UIDs for use in naming the Theano compiledir."
    " Optional, but should be same length as --gpu-workers if specified." )
  nnsearch_group.add_argument( "--model-dir", type=str, default=None, help="Location to save/load models (default: same as -output)" )
  nnsearch_group.add_argument( "--purge-models", action="store_true", help="Delete all stored models before starting search." )
  nnsearch_group.add_argument( "--log-gpu-energy", action="store_true", help="Log GPU energy usage during model evaluation." )
  nnsearch_group.add_argument( "--algorithm", type=str, choices=["mcts", "ce", "bandit"], help="Search algorithm" )
  objective_group = parser.add_argument_group( "objective" )
  objective_group.add_argument( "--objective", type=str, choices=["accuracy", "linear_accuracy_energy", "accuracy_per_energy"],
    default="accuracy", help="Optimization objective" )
  objective_group.add_argument( "--obj-energy-scale", type=float, default=1e-3, help="Scale for energy component of objective." )
  objective_group.add_argument( "--obj-energy-weight", type=float, default=0.5, help="Energy weight in linear_accuracy_energy [0,1]" )
  bandit_group = parser.add_argument_group( "bandit" )
  bandit_group.add_argument( "--bandit-rule", type=str, choices=["ucb1", "ucb-sqrt", "seq-halve"], default="ucb1", help="Bandit rule" )
  bandit_group.add_argument( "--bandit-ucb-c", type=float, default=1.0, help="Exploration constant for Ucb-type rules." )
  bandit_group.add_argument( "--bandit-nsteps", type=int, default=1, help="Number of arm pulls" )
  bandit_group.add_argument( "--bandit-eval-epochs", type=int, default=1,
    help="Number of epochs to train network before evaluating." )
  bandit_group.add_argument( "--bandit-warmup", type=int, default=0,
    help="Number of epochs to train every network before starting search." )
  ce_group = parser.add_argument_group( "cross-entropy" )
  ce_group.add_argument( "--ce-generations", type=int, help="Number of generations" )
  ce_group.add_argument( "--ce-generation-size", type=int, help="Size of generation" )
  ce_group.add_argument( "--ce-mix-factor", type=float,
    help="Mixing factor [0,1] for combining old policy with improved policy."
    " Larger values make the policy change more quickly." )
  ce_group.add_argument( "--ce-weight", type=str, choices=["proportional", "best"], default="proportional",
    help="Weight function" )
  ce_group.add_argument( "--ce-best-fraction", type=float, default=0.1,
    help="'Elite' fraction for 'best' weight function." )
  ce_group.add_argument( "--ce-feature-space", type=str, choices=["WistubaA", "Wistuba_Test"],
    help="Policy feature space" )
  ce_group.add_argument( "--ce-eval-epochs", type=int, default=1,
    help="Number of epochs to train network before evaluating." )
  ce_group.add_argument( "--ce-cumulative-training", action="store_true",
    help="Enable cumulative training. Network architectures will be trained some more each time they are proposed." )
  ce_group.add_argument( "--ce-shallowness", type=float, default=0.5,
    help="Probability mass [0,1] reserved for actions that favor shallow networks in the initial policy." )
  # 2nd positional argument (`arch`) is mandatory, but we're not using it
  if len(sys.argv) > 1:
    argv = [sys.argv[1], ""] + sys.argv[2:]
  else:
    argv = sys.argv
  args = parser.parse_args( argv )
  print( "Initial args: {}".format( args ) )
  # Default arguments
  if args.model_dir is None:
    args.model_dir = args.output
  if args.objective != "accuracy" and args.algorithm != "bandit":
    raise ValueError( "Objective {} only supported for --algorithm=bandit".format( args.objective ) )
  
  logfile = os.path.join( args.output, "output_main.log" )
  logging.basicConfig( filename=logfile, filemode="w",
                       format=nnspace.wistuba.NeuralNetworkSearchSpace.log_format, level=args.level )
  log = logging.getLogger( __name__ )
  log.info( "Main process is alive (pid: %s)", os.getpid() )
  
  print( "[Main] Environment:" )
  for k in sorted( os.environ ):
    print( k + ": " + os.environ[k] )
  
  # Root of the RNG hierarchy
  root_rng = random.Random( args.seed )
  
  if args.json is not None:
    print( "Parsing json..." )
    with open( args.json ) as fin:
      json_dict = json.load( fin )
    
    print( "Initializing search space..." )
    # TODO: Hardcoded index `0`
    search_space = nnspec.parse_json_search_space_list( json_dict["search_space"] )[0]
    input = search_space["input"]
    output = search_space["output"]
    layers = search_space["layers"]
  elif args.arch_list is not None:
    print( "Parsing arch-list..." )
    with open( args.arch_list ) as fin:
      models = [nnspec.LayerSpec.from_arch_string(line.rstrip()) for line in fin]
    for m in models:
      print( "'" + str(m) + "'" )
    # sys.exit( 0 )
    
  storage = nnsearch.storage.ModelDirectory( args.model_dir, ".tgz" )
  backend = nnsearch.backend.architect.FeedForward
  parameters = args
  gpu_workers = [int(id) for id in args.gpu_workers.split( "," )] if args.gpu_workers else []
  gpu_names = args.gpu_names.split( "," ) if args.gpu_names else None
  
  if args.purge_models:
    storage.delete_models()
  
  print( "Beginning search..." )
  if args.algorithm == "mcts":
    raise NotImplementedError() # TODO:
    search = SearchAlgorithm.from_json( json_dict["search_algorithm"] )
    sn0 = search( model )
    nnsearch.mcts.print_tree( sn0 )
  elif args.algorithm == "bandit":
    # i-1-28-28(alias-input):conv-12-3-1-1(alias-c0):act-relu:[+input]conv-24-3-1-1:act-relu:[+input,c0]conv-48-3-1-1:act-relu:fc-1024:l-10
    pool = nnsearch.worker.Pool( storage, backend, parameters,
      nworkers=args.nworkers, gpu_workers=gpu_workers, gpu_names=gpu_names )
    gpu_logger = None
    if args.log_gpu_energy:
      gpu_logger = nnsearch.gpu.GpuLogger( args.output )
    if args.bandit_rule == "ucb1":
      bandit = nnsearch.bandit.Bandit( nnsearch.bandit.Ucb1( c=args.bandit_ucb_c ), len(models) )
    elif args.bandit_rule == "ucb-sqrt":
      bandit = nnsearch.bandit.Bandit( nnsearch.bandit.UcbSqrt( c=args.bandit_ucb_c ), len(models) )
    elif args.bandit_rule == "seq-halve":
      bandit = nnsearch.bandit.SequentialHalvingBandit( len(models), args.bandit_nsteps )
    bandit_search = nnsearch.flat.BanditSearch( models, bandit, pool, gpu_logger=gpu_logger )
    if args.objective == "accuracy":
      value_fn = lambda perf: perf.accuracy.validation
    elif args.objective == "accuracy_per_energy":
      value_fn = lambda perf: objective.accuracy_per_energy(
        perf.accuracy.validation, perf.energy.total, args.obj_energy_scale )
    elif args.objective == "linear_accuracy_energy":
      value_fn = lambda perf: objective.linear_accuracy_energy(
        perf.accuracy.validation, perf.energy.total, args.obj_energy_weight, args.obj_energy_scale )
    rng = random.Random( root_rng.randrange(sys.maxsize) )
    
    try:
      bandit_search.search( rng, args.bandit_nsteps, value_fn,
        nepoch=args.bandit_eval_epochs, warmup=args.bandit_warmup )
    finally:
      if gpu_logger is not None:
        gpu_logger.shutdown()
      pool.shutdown()
  elif args.algorithm == "ce":
    model = nnspace.wistuba.NeuralNetworkSearchSpace( input, output, layers )
    pool = nnsearch.worker.Pool( storage, backend, parameters,
      nworkers=args.nworkers, gpu_workers=gpu_workers, gpu_names=gpu_names )
    rng = random.Random( root_rng.randrange(sys.maxsize) )
    if args.ce_feature_space == "WistubaA":
      feature_space = nnspace.wistuba.WistubaFeatureSpaceA()
    elif args.ce_feature_space == "Wistuba_Test":
      feature_space = nnspace.wistuba.WistubaFeatureSpace_Test()
    
    gpu_logger = None
    if args.log_gpu_energy:
      gpu_logger = nnsearch.gpu.GpuLogger( args.output )
    try:
      ce_model = nnspace.wistuba.CrossEntropyTabularModel( rng, model, pool,
        feature_space=feature_space, eval_epochs=args.ce_eval_epochs,
        mix_factor=args.ce_mix_factor, shallowness=args.ce_shallowness,
        cumulative_training=args.ce_cumulative_training,
        gpu_logger=gpu_logger )
      cross_entropy_search( ce_model, generations=args.ce_generations, 
        generation_size=args.ce_generation_size )
    finally:
      pi = ce_model.policy()
      # pi: s -> a -> prob
      for (k, v) in pi.items():
        p = list(v.values())
        log.info( "%s: %s", k, p )
      
      if gpu_logger is not None:
        gpu_logger.shutdown()
      pool.shutdown()
