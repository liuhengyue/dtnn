from collections import OrderedDict
import timeit
import torch
import torch.nn as nn

# a simple network

modules = OrderedDict()
n_conv = 10
n_linear = 2
modules["conv_0"] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
for i in range(1, n_conv+1):
    modules["conv_{}".format(i)] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
    modules["relu_{}".format(i)] = nn.ReLU()

# for i in range(n_linear):
#     modules["fc_{}".format(i)] = nn.Linear(512, 256)


rand_net = nn.Sequential(modules)

def init_normal(m):
    if type(m) == nn.Conv2d:
        nn.init.uniform_(m.weight)

def init_zero(m):
    if type(m) == nn.Conv2d:
        nn.init.zeros_(m.weight)


def experiment(model, init_function, input, n_trials, cuda=False):
    device = "gpu" if cuda else "cpu"
    print("Runnning on {} with {} experiments.".format(device, n_trials))
    model.apply(init_function)
    model.eval()
    if cuda:
        model = model.cuda()
        input = input.cuda()
    start = timeit.default_timer()
    for _ in range(n_trials):
        _ = model(input)
    stop = timeit.default_timer()
    print('{} conv layers with {} weights - forward time: {:.4f}'.format(n_conv + 1, init_function.__name__, (stop - start) / n_trials))

    start = timeit.default_timer()
    for _ in range(n_trials):
        output = model(input)
        output.backward
    stop = timeit.default_timer()
    print('{} conv layers with {} weights - forward + backward time: {:.4f}'.format(n_conv + 1, init_function.__name__,
                                                                         (stop - start) / n_trials))

input = torch.randn((1, 3, 32, 32))
experiment(rand_net, init_normal, input, 1000, cuda=False)
experiment(rand_net, init_zero, input, 1000, cuda=False)
experiment(rand_net, init_normal, input, 1000, cuda=True)
experiment(rand_net, init_zero, input, 1000, cuda=True)
