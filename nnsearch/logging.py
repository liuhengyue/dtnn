import logging
import os
import subprocess

def git_revision():
  def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH']:
      v = os.environ.get(k)
      if v is not None:
        env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
    return out

  try:
    out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
    revision = out.strip().decode('ascii')
  except OSError:
    revision = "Unknown"
  return revision

def log_level_from_string( s ):
  level = getattr(logging, s)
  if not isinstance(level, int):
    raise ValueError( "Definitely not a log level name" )
  return level

def add_log_level(levelName, levelNum, methodName=None):
  """
  Comprehensively adds a new logging level to the `logging` module and the
  currently configured logging class.

  `levelName` becomes an attribute of the `logging` module with the value
  `levelNum`. `methodName` becomes a convenience method for both `logging`
  itself and the class returned by `logging.getLoggerClass()` (usually just
  `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
  used.

  To avoid accidental clobberings of existing attributes, this method will
  raise an `AttributeError` if the level name is already an attribute of the
  `logging` module or if the method name is already present 

  Example
  -------
  >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
  >>> logging.getLogger(__name__).setLevel("TRACE")
  >>> logging.getLogger(__name__).trace('that worked')
  >>> logging.trace('so did this')
  >>> logging.TRACE
  5

  """
  if methodName is None:
    methodName = levelName.lower()

  if hasattr(logging, levelName):
    raise AttributeError('{} already defined in logging module'.format(levelName))
  if hasattr(logging, methodName):
    raise AttributeError('{} already defined in logging module'.format(methodName))
  if hasattr(logging.getLoggerClass(), methodName):
    raise AttributeError('{} already defined in logger class'.format(methodName))

  # This method was inspired by the answers to Stack Overflow post
  # http://stackoverflow.com/q/2183233/2988730, especially
  # http://stackoverflow.com/a/13638084/2988730
  def logForLevel(self, message, *args, **kwargs):
    if self.isEnabledFor(levelNum):
      self._log(levelNum, message, args, **kwargs)
  def logToRoot(message, *args, **kwargs):
    logging.log(levelNum, message, *args, **kwargs)

  logging.addLevelName(levelNum, levelName)
  setattr(logging, levelName, levelNum)
  setattr(logging.getLoggerClass(), methodName, logForLevel)
  setattr(logging, methodName, logToRoot)
