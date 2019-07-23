import argparse
import shlex

class MyArgumentParser(argparse.ArgumentParser):
  def convert_arg_line_to_args( self, arg_line ):
    return shlex.split( arg_line )

def range_check( minval, maxval, min_exclusive=False, max_exclusive=False ):

  class RangeChecker(argparse.Action):
    def __init__( self, *args, **kwargs ):
      super().__init__( *args, **kwargs )
      self.min = minval
      self.max = maxval
      self.min_exclusive = min_exclusive
      self.max_exclusive = max_exclusive

    def __call__( self, parser, namespace, values, option_string=None ):
      if self.min is not None:
        if (self.min_exclusive and values <= self.min) or values < self.min:
          op = ">" if self.min_exclusive else ">="
          parser.error( "Value of {} must be {} {}".format(option_string, op, self.min) )
      if self.max is not None:
        if (self.max_exclusive and values >= self.max) or values > self.max:
          op = "<" if self.max_exclusive else "<="
          parser.error( "Value of {} must be {} {}".format(option_string, op, self.max) )
      setattr(namespace, self.dest, values)
      
  return RangeChecker
