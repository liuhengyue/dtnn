import ast
import logging

import torch

import torchvision

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

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

# FIXME: Shouldn't take arguments as an `args` instance because that couples
# the implementation to argument parsing
  
def densenet169( args, dataset ):
  assert args.dataset == "imagenet"
  # These settings are hardcoded in the PyTorch implementation
  assert args.densenet_bottleneck
  assert args.densenet_compression == 0.5
  nlayers = [int(t) for t in args.gated_densenet.split(",")]
  pretrained = torchvision.models.densenet169( pretrained=True )
  log.info( "pretrained model:\n%s", pretrained )
  
  # Indexing information for our gated implementation
  names = {}
  ntrans = 4
  stride = 1 + ntrans # 1 for DenseBlock
  nblocks = len(nlayers)
  ninput = 1
  ndense = 1
  nbody = nblocks*ndense + (nblocks - 1)*ntrans
  noutput = 3
  # Start of each "stage": input -> body -> output -> classifier
  input_idx = 0
  body_idx = ninput
  output_idx = body_idx + nbody
  classifier_idx = output_idx + noutput
  
  # In these functions, 'pname' is the PyTorch name prefix, and 'gname' is
  # the gated network name prefix. Here "prefix" means the part of the full
  # name that is context-specific.
  
  def input( pname, gname ):
    layers = ["conv", "norm", "relu", "pool"]
    for (i, p) in enumerate(layers):
      names["{}{}0".format(pname, p)] = "{}{}".format(gname, i)
  
  def denseblock( n, pname, gname ):
    for j in range(n):
      denselayer( "{}denselayer{}.".format(pname, j+1), 
                  "{}{}.".format(gname, j) )
  
  def denselayer( pname, gname ):
    layers = ["norm", "relu", "conv"]
    k = len(layers)
    for i in range(2):
      for (j, p) in enumerate(layers):
        names["{}{}{}".format(pname, p, i+1)] = "{}{}".format(gname, i*k + j)
  
  # 'idx' is index of first Module of transition block in the gated network
  def transition( idx, pname, gname ):
    for (i, layer) in enumerate(["norm", "relu", "conv", "pool"]):
      names["{}{}".format(pname, layer)] = "{}{}".format(gname, idx + i)
  
  def output( pname, gname ):
    # Final ReLU and AvgPool are not distinct layers in the PyTorch model
    names["{}norm{}".format(pname, nblocks+1)] = "{}{}".format(gname, output_idx)
    
  def classifier( pname, gname ):
    names["{}classifier".format(pname)] = "{}{}".format(gname, classifier_idx)
    
  input( "features.", "fn.0." )
  for (i, n) in enumerate(nlayers):
    dense_block_idx = ninput + stride*i
    denseblock( n, "features.denseblock{}.".format(i+1), 
                   "fn.{}.components.".format(dense_block_idx) )
    if i < (nblocks - 1):
      transition( dense_block_idx + ndense,
                  "features.transition{}.".format(i+1), "fn." )
  output( "features.", "fn." )
  classifier( "", "fn." ) # 'classifier' is not under 'features'
  
  for k in sorted(names):
    log.debug( "%s -> %s", k, names[k] )
  
  original = pretrained.state_dict()
  translated = {}
  for pname in original:
    # The final name in the chain in the state_dict is the name of the
    # parameter within the layer (e.g. "weight"). We translate the prefix
    # then put the parameter name back at the end.
    tokens = pname.split( "." )
    prefix = ".".join( tokens[:-1] )
    rename = names[prefix] + "." + tokens[-1]
    log.info( "state_dict: %s -> %s", pname, rename )
    translated[rename] = original[pname]
  return translated
    
def resnext50a( args, dataset ):
  # See 'resnext50-translation.txt' for detailed description of translation
  
  def resnext_c_to_a( conv, bn, ngroups ):
    def path( i ):
      # Note: conv weights are indexed [out dim, in dim, ...]
      a, b, c = conv
      aout = a.shape[0] // ngroups
      bout = b.shape[0] // ngroups
      cin = c.shape[1] // ngroups
      
      # ai: output is split into groups
      ai = a[i*aout:(i+1)*aout, :, :, :].clone()
      # bi: grouped conv has the right number of inputs; split its
      # output into groups
      bi = b[i*bout:(i+1)*bout, :, :, :].clone()
      # ci: split input into groups, preserve output size
      ci = c[:,   i*cin:(i+1)*cin, :, :].clone()
      uconv = [ai, bi, ci]
      
      ubn = []
      for w in bn:
        chunk = w.shape[0] // ngroups
        ubn.append( w[i*chunk:(i+1)*chunk] )
      return (uconv, ubn)
      
    return [path(i) for i in range(ngroups)]

  assert args.skip_connection_batchnorm

  original = torch.load( os.path.join( args.input, "resnext_50_32x4d.pth" ) )
  translated = {}
  
  def bn_defaults( key ):
    translated["{}.num_batches_tracked".format(key)] = (
      torch.tensor( 0, dtype=torch.long ))
  
  # FIXME: copy-pasted architecture spec parsing; should pass model instance
  # as a parameter or something
  
  # Input module
  defaults = resnext.defaults(dataset)
  expansion = (args.resnext_expansion if args.resnext_expansion is not None 
               else defaults.expansion)
  # Rest of network
  stages = eval(args.gated_resnext)
  stages = [resnext.ResNeXtStage(*t, expansion) for t in stages]
  
  nstages = len(stages)
  nresnext_layers = sum( s.nlayers for s in stages )
  norig_top_level = nstages + 7
  bn_params = ["weight", "bias", "running_mean", "running_var"]
  
  # Input module: conv -> bn -> relu -> maxpool
  translated["fn.0.0.weight"] = original["0.weight"]
  for p in bn_params:
    translated["fn.0.1.{}".format(p)] = original["1.{}".format(p)]
  bn_defaults( "fn.0.1" )
  
  # ResNeXt stage
  ti = 1 # index in translated dict
  for i in range(nstages):
    ngroups = stages[i].ncomponents
    assert ngroups == 32
    si = i + 4 # input module has 4 layers
    for li in range(stages[i].nlayers):
      # Split grouped weights into ngroups independent paths
      opre = "{}.{}.0".format(si, li)
      conv_keys = [
        opre + ".0.0.0.weight", opre + ".0.0.3.weight", opre + ".0.1.weight"]
      # Note: The third BN happens after aggregation in model (a), thus does
      # not need to be split
      bn_keys = ["{}.0.0.{}.{}".format(opre, idx, p)
                 for idx in [1, 4] for p in bn_params]
      conv = [original[k] for k in conv_keys]
      bn = [original[k] for k in bn_keys]
      ungrouped = resnext_c_to_a( conv, bn, ngroups=ngroups )
      # Translate names
      tpre = "fn.{}.components".format(ti)
      for g in range(ngroups):
        uconv, ubn = ungrouped[g]
        bni = 0
        # conv
        translated["{}.{}.0.weight".format(tpre, g)] = uconv[0]
        # bn
        for p in bn_params:
          translated["{}.{}.1.{}".format(tpre, g, p)] = ubn[bni]
          bni += 1
        bn_defaults( "{}.{}.1".format(tpre, g) )
        # conv
        translated["{}.{}.3.weight".format(tpre, g)] = uconv[1]
        # bn
        for p in bn_params:
          translated["{}.{}.4.{}".format(tpre, g, p)] = ubn[bni]
          bni += 1
        bn_defaults( "{}.{}.4".format(tpre, g) )
        # conv
        translated["{}.{}.6.weight".format(tpre, g)] = uconv[2]
      # Final batchnorm after the aggregation
      for p in bn_params:
        translated["fn.{}.aggregate_batchnorm.{}".format(ti, p)] = (
          original["{}.0.2.{}".format(opre, p)])
      bn_defaults( "fn.{}.aggregate_batchnorm".format(ti) )
      # Skip-connection downsampling layer, if present
      downsample = "{}.1.0.weight".format(opre) in original
      if downsample:
        translated["fn.{}.downsample.0.weight".format(ti)] = (
          original["{}.1.0.weight".format(opre)])
        for p in bn_params:
          translated["fn.{}.downsample.1.{}".format(ti, p)] = (
            original["{}.1.1.{}".format(opre, p)])
        bn_defaults( "fn.{}.downsample.1".format(ti) )
      # Our version of the model flattens the stage/layer distinction
      ti += 1
  
  # Output module: AvgPool2d -> FullyConnected
  ti += 1 # skip pooling layer
  assert ti == 18
  translated["fn.{}.weight".format(ti)] = (
    original["{}.1.weight".format(norig_top_level - 1)])
  translated["fn.{}.bias".format(ti)] = (
    original["{}.1.bias".format(norig_top_level - 1)])
  for k, v in translated.items():
    log.info( "state_dict: %s : %s", k, v.shape )
  return translated
  
def resnext50c( args, dataset ):
  # See 'resnext50-translation.txt' for detailed description of translation

  assert args.skip_connection_batchnorm

  original = torch.load( os.path.join( args.input, "resnext_50_32x4d.pth" ) )
  translated = {}
  
  def linear( tkey, okey ):
    translated["{}.weight".format(tkey)] = original["{}.weight".format(okey)]
    if "{}.bias".format(okey) in original:
      translated["{}.bias".format(tkey)] = original["{}.bias".format(okey)]
  
  bn_params = ["weight", "bias", "running_mean", "running_var"]    
  def batchnorm( tkey, okey ):
    for p in bn_params:
      translated["{}.{}".format(tkey, p)] = original["{}.{}".format(okey, p)]
    translated["{}.num_batches_tracked".format(tkey)] = (
      torch.tensor(0, dtype=torch.long))
  
  # FIXME: copy-pasted architecture spec parsing; should pass model instance
  # as a parameter or something
  
  # Input module
  defaults = resnext.defaults(dataset)
  expansion = (args.resnext_expansion if args.resnext_expansion is not None 
               else defaults.expansion)
  # Rest of network
  stages = eval(args.gated_resnext)
  stages = [resnext.ResNeXtStage(*t, expansion) for t in stages]
  
  nstages = len(stages)
  nresnext_layers = sum( s.nlayers for s in stages )
  norig_top_level = nstages + 7
  
  # Input module: conv -> bn -> relu -> maxpool
  linear( "fn.0.0", "0" )
  batchnorm( "fn.0.1", "1" )
  
  # ResNeXt stage
  ti = 1 # index in translated dict
  for i in range(nstages):
    ngroups = stages[i].ncomponents
    assert ngroups == 32
    si = i + 4 # input module has 4 layers
    for li in range(stages[i].nlayers):
      opre = "{}.{}.0".format(si, li)
      
      # Residual path
      linear( "fn.{}.residual.0".format(ti), "{}.0.0.0".format(opre) )
      batchnorm( "fn.{}.residual.1".format(ti), "{}.0.0.1".format(opre) )
      linear( "fn.{}.residual.3".format(ti), "{}.0.0.3".format(opre) )
      batchnorm( "fn.{}.residual.4".format(ti), "{}.0.0.4".format(opre) )
      linear( "fn.{}.residual.6".format(ti), "{}.0.1".format(opre) )
      batchnorm( "fn.{}.residual.7".format(ti), "{}.0.2".format(opre) )
      
      # Skip-connection downsampling layer, if present
      skip = "{}.1.0.weight".format(opre) in original
      if skip:
        linear( "fn.{}.skip.0".format(ti), "{}.1.0".format(opre) )
        batchnorm( "fn.{}.skip.1".format(ti), "{}.1.1".format(opre) )
      
      # Our version of the model flattens the stage/layer distinction
      ti += 1
  
  # Output module: AvgPool2d -> FullyConnected
  ti += 1 # skip pooling layer
  assert ti == 18
  linear( "fn.{}".format(ti), "{}.1".format(norig_top_level - 1) )
  for k, v in translated.items():
    log.info( "state_dict: %s : %s", k, v.shape )
  return translated

def resnet50( network, args, dataset ):
  pretrained = torchvision.models.resnet50( pretrained=True )
  log.info( "pretrained model:\n%s", pretrained )
  return _resnet( pretrained, network, args, dataset )
  
def resnet101( network, args, dataset ):
  pretrained = torchvision.models.resnet101( pretrained=True )
  log.info( "pretrained model:\n%s", pretrained )
  return _resnet( pretrained, network, args, dataset )
  
def _resnet( pretrained, network, args, dataset ):
  assert args.dataset == "imagenet"
  
  # Rest of network
  stages = ast.literal_eval(args.blockdrop_resnet)
  stages = [resnet.ResNetStageSpec(*t) for t in stages]
  
  original = pretrained.state_dict()
  translated = network.state_dict()
  
  names = {k: None for k in translated}
  # print( names )
  
  bn = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
  weight = ["weight", "bias"]
  def params( old, new, fields ):
    for f in fields:
      names["{}.{}".format(old, f)] = "{}.{}".format(new, f)
  
  # Input
  params( "fn.0.0", "conv1", weight )
  params( "fn.0.1", "bn1", bn )
  
  # ResNet blocks
  for (si, stage) in enumerate(stages):
    for bi in range(stage.nblocks):
      ii = (si+1, bi)
      params( "fn.{}.{}.residual.0".format(*ii),
              "layer{}.{}.conv1".format(*ii), weight )
      params( "fn.{}.{}.residual.1".format(*ii),
              "layer{}.{}.bn1".format(*ii), bn )
      params( "fn.{}.{}.residual.3".format(*ii),
              "layer{}.{}.conv2".format(*ii), weight )
      params( "fn.{}.{}.residual.4".format(*ii),
              "layer{}.{}.bn2".format(*ii), bn )
      params( "fn.{}.{}.residual.6".format(*ii),
              "layer{}.{}.conv3".format(*ii), weight )
      params( "fn.{}.{}.residual.7".format(*ii),
              "layer{}.{}.bn3".format(*ii), bn )
      params( "fn.{}.{}.skip.0".format(*ii),
              "layer{}.{}.downsample.0".format(*ii), weight )
      params( "fn.{}.{}.skip.1".format(*ii),
              "layer{}.{}.downsample.1".format(*ii), bn )

  # Output
  params( "fn.5.1", "fc", weight )
  
  for k in sorted(translated):
    log.verbose( "pretrained.translated: {} -> {}".format(k, names[k]) )
    translated[k] = original[names[k]]
  return translated
  
  