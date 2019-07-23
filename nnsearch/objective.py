from collections import namedtuple

Accuracy = namedtuple( "Accuracy", ["validation", "testing"] )
Energy   = namedtuple( "Energy",   ["total", "active"] )

Performance = namedtuple( "Performance", ["accuracy", "energy"] )

def accuracy_per_energy( accuracy, energy, energy_scale=1e-4 ):
  return accuracy / (energy_scale * energy)
  
def linear_accuracy_energy( accuracy, energy, energy_weight=0.5, energy_scale=1e-4 ):
  assert( 0 <= energy_weight )
  assert( energy_weight <= 1 )
  return (1.0 - energy_weight) * accuracy - energy_weight * energy_scale * energy
