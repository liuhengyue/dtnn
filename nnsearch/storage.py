import hashlib
import logging
import os
import os.path
import re

class ModelDirectory:
  """ Simple interface to a file system directory that stores models. Stored
  models are queries by their "spec string". This class only manages the file
  names. Loading and saving is left to the backend implementation.
  """
  
  def __init__( self, dir, suffix="" ):
    """
    Parameters:
      `dir` : Directory to store the models. Must exist.
      `suffix` : Optional string to append to end of all generated filenames.
    """
    self._dir = dir
    self._suffix = suffix
    
  def get_filename( self, s ):
    """
    Parameters:
      `s` : String representation of model.
    
    Returns:
      The filename corresponding to the model.
    """
    h = hashlib.md5( s.encode("utf-8") )
    return os.path.join( self._dir, h.hexdigest() ) + self._suffix
    
  def contains( self, s ):
    """ Returns True if a model file exists for `s`.
    """
    name = self.get_filename( s )
    return os.path.isfile( name )

  def delete_models( self ):
    log = logging.getLogger( __name__ )
    pattern = re.compile( r"^[a-f0-9]{32}" + re.escape( self._suffix ) + r"$" )
    for f in os.listdir( self._dir ):
      if re.match( pattern, f ):
        log.info( "Deleting '{}'".format( f ) )
        fpath = os.path.join( self._dir, f )
        os.remove( fpath )