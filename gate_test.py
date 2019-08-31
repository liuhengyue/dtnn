import torch
from modules.utils import *
import nnsearch.pytorch.gated.strategy as strategy

gate_control = uniform_gate(0.5)

ncomponents = 10

count = strategy.UniformCount(ncomponents)
                        # use all components
                        # count = strategy.ConstantCount(stage.ncomponents, stage.ncomponents)
gate = strategy.NestedCountGate(ncomponents, count, gate_during_eval=True)

print(gate)

input = torch.randn(2, 16, 32, 32)

nactive = count(input)

print(nactive)

output = gate(input)

print(output)

proportion_count = strategy.ProportionToCount(0, ncomponents)

gate = strategy.NestedCountFromUGate(ncomponents, proportion_count, gate_during_eval=True)

gate.set_control(torch.tensor(0.0))

print(gate(input))
