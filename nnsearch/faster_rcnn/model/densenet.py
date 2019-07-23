import nnsearch.nnspec
from nnsearch.nnspec import LayerSpec

def densenet( input, output, nblocks, growth_rate, block_sizes, bottleneck=False ):
  assert( type(input) == nnsearch.nnspec.InputLayerSpec )
  layers = [input]
  input_h = input.height()
  input_w = input.width()
  for i in range(nblocks):
    if i > 0:
      input_size = block_sizes[i-1] * growth_rate
      layers.append( LayerSpec.Sequence.value( 
        LayerSpec.BatchNorm.value(), LayerSpec.Activation.value("relu"),
        LayerSpec.Convolution.value(input_size, 1),
        LayerSpec.SquarePool.value("average_exc_pad", 2, 2) ) )
    block = densenet_block( growth_rate, block_sizes[i], i, bottleneck=bottleneck )
    layers.append( block )
    assert( input_h % 2 == 0 )
    assert( input_w % 2 == 0 )
    input_h //= 2
    input_w //= 2
  layers.append( LayerSpec.Pool.value("average_exc_pad", input_h, input_w, input_h, input_w) )
  layers.append( output )
  return LayerSpec.Sequence.value( *layers )
  
def densenet_block( growth_rate, size, block_id, bottleneck ):
  block = []
  for i in range(size):
    layer = []
    alias = "b{}c{}".format( block_id, i )
    if bottleneck:
      layer.append( LayerSpec.BatchNorm.value() )
      layer.append( LayerSpec.Activation.value("relu") )
      layer.append( LayerSpec.Convolution.value(growth_rate, 1) )
    layer.append( LayerSpec.BatchNorm.value() )
    layer.append( LayerSpec.Activation.value("relu") )
    layer.append( LayerSpec.Convolution.value(growth_rate, 3) )
    seq = LayerSpec.Sequence.value( *layer, alias=alias, inputs=block[:-1] )
    block.append( seq )
  return LayerSpec.Sequence.value( *block )
  
if __name__ == "__main__":
  input = LayerSpec.Input.value( 3, 32, 32 )
  output = LayerSpec.Sequence.value( LayerSpec.FullyConnected.value( 1000 ), LayerSpec.Output.value( 10 ) )
  # nn = densenet_block( 8, 4, 0, bottleneck=True )
  nn = densenet( input, output, nblocks=3, growth_rate=12, block_sizes=[4]*3, bottleneck=True )
  print( nn.arch_string() )
