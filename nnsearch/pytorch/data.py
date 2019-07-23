import os.path

import torchvision
import torchvision.transforms as transforms

class MnistDataset:
  name = "mnist"

  def preprocess( self, method ):
    if method is not None:
      raise ValueError( "Unknown method '{}'".format( method ) )
    
  def load( self, root, train, download=True ):
    return torchvision.datasets.MNIST(
      root=root, train=train, transform=self.transform( train ), download=download )
  
  @staticmethod
  def _normalize( x ):
    # torchvision MNIST is in [0, 1]
    return x
  
  def transform( self, train ):
    return transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda( MnistDataset._normalize ) ])
  
  @property
  def nclasses( self ):
    return 10
    
  @property
  def in_shape( self ):
    return (1, 28, 28)
  
  @property
  def class_names( self ):
    return tuple( map( str, range(self.nclasses) ) )
    
class FashionMnistDataset:
  name = "fashion_mnist"

  def preprocess( self, method ):
    if method is not None:
      raise ValueError( "Unknown method '{}'".format( method ) )
    
  def load( self, root, train, download=True ):
    return torchvision.datasets.FashionMNIST(
      root=root, train=train, transform=self.transform( train ), download=download )
  
  @staticmethod
  def _normalize( x ):
    # torchvision MNIST is in [0, 1]
    return x
  
  def transform( self, train ):
    return transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda( MnistDataset._normalize ) ])
  
  @property
  def nclasses( self ):
    return 10
    
  @property
  def in_shape( self ):
    return (1, 28, 28)
  
  @property
  def class_names( self ):
    return ("tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot")
    
class Cifar10Dataset:
  name = "cifar10"

  def __init__( self ):
    self._transform  = [transforms.ToTensor(), transforms.Lambda( Cifar10Dataset._normalize )]
    self._preprocess = []
    
  def preprocess( self, method ):
    if method is None:
      self._preprocess = []
    elif "resnet" == method:
      self._preprocess = [transforms.Pad(4), transforms.RandomHorizontalFlip(0.5), transforms.RandomCrop(32)]
    else:
      raise ValueError( "Unknown method '{}'".format(method) )
    
  def load( self, root, train, download=True ):
    return torchvision.datasets.CIFAR10(
      root=root, train=train, transform=self.transform( train ), download=download )
  
  @staticmethod
  def _normalize( x ):
    # torchvision CIFAR is in [0, 1]
    return x
  
  def transform( self, train ):
    if train:
      return transforms.Compose( self._preprocess + self._transform )
    else:
      return transforms.Compose( self._transform )
  
  @property
  def nclasses( self ):
    return 10
    
  @property
  def in_shape( self ):
    return (3, 32, 32)
  
  @property
  def class_names( self ):
    return ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    
class ImageNetDataset:
  name = "imagenet"

  def __init__( self ):
    # see: pytorch/examples/imagenet/main.py
    self._transform  = [transforms.ToTensor(), transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    self.preprocess()
    
  def preprocess( self, size=224, method="pytorch" ):
    if method is None:
      self._preprocess_train = [
        transforms.Resize(256), transforms.CenterCrop(224)]
      self._preprocess_eval  = [
        transforms.Resize(256), transforms.CenterCrop(224)]
    elif "pytorch" == method:
      # see: pytorch/examples/imagenet/main.py
      self._preprocess_train = [
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
      self._preprocess_eval  = [
        transforms.Resize(256), transforms.CenterCrop(224)]
    else:
      raise ValueError( "Unknown method '{}'".format(method) )
    
  def load( self, root, train ):
    sub = "train" if train else "val"
    return torchvision.datasets.ImageFolder(
      root=os.path.join( root, sub ), transform=self.transform( train ) )
  
  def transform( self, train ):
    if train:
      return transforms.Compose( self._preprocess_train + self._transform )
    else:
      return transforms.Compose( self._preprocess_eval  + self._transform )
  
  @property
  def nclasses( self ):
    return 1000
    
  @property
  def in_shape( self ):
    return (3, 224, 224)
  
  @property
  def class_names( self ):
    return tuple( map( str, range(self.nclasses) ) )
    
class SvhnDataset:
  name = "svhn"
  
  def __init__( self ):
    self._transform  = [
      transforms.ToTensor(),
      transforms.Lambda( SvhnDataset._normalize )]
    self._preprocess = []
    
  def preprocess( self, method ):
    if method is None:
      self._preprocess = []
    elif "resnet" == method:
      self._preprocess = [transforms.Pad(4), transforms.RandomHorizontalFlip(0.5), transforms.RandomCrop(32)]
    else:
      raise ValueError( "Unknown method '{}'".format(method) )
    
  def load( self, root, train, download=True ):
    return torchvision.datasets.CIFAR10(
      root=root, train=train, transform=self.transform( train ), download=download )
  
  @staticmethod
  def _normalize( x ):
    # torchvision SVHN is in [0, 1]
    return x
  
  def transform( self, train ):
    if train:
      return transforms.Compose( self._preprocess + self._transform )
    else:
      return transforms.Compose( self._transform )
  
  @property
  def nclasses( self ):
    return 10
    
  @property
  def in_shape( self ):
    return (3, 32, 32)
  
  @property
  def class_names( self ):
    return tuple( map( str, range(self.nclasses) ) )
    
datasets = {t.name: t() for t in [
  Cifar10Dataset, FashionMnistDataset, ImageNetDataset, MnistDataset, SvhnDataset
]}
