import os
import random
import numpy
import math
import glob

import re
import logging

import torch
import torch.optim as optim

import numpy as np

import torch.nn.functional as fn
from torch.nn.parameter import Parameter
import torch.nn.init as init

import gym
import gym.spaces

from network.gated_c3d import GatedC3D
from network.gated_c3d import GatedStage
from nnsearch.pytorch.gated import strategy
from nnsearch.pytorch.parameter import *
from nnsearch.pytorch.rl import *
# todo: eval function has some logger stuff
from nnsearch.pytorch.rl.env.solar import SolarEpisodeLogger
from nnsearch.pytorch.rl.policy import *
from nnsearch.pytorch.rl.dqn import *
from nnsearch.statistics import *
from nnsearch.pytorch.gated.module import BlockGatedConv3d, BlockGatedConv2d, BlockGatedFullyConnected
from network.demo_model import GestureNet

from bandit_net import ContextualBanditNet
import shape_flop_util as util
import model_util

##################################################################################################

class GatedNetworkApp:
    def __init__(self, argument_parser):
        pass

    def init_gated_network_parameters(self, network, from_file=None):
        if from_file is not None:
            skip = model_util.is_gate_param
            self.checkpoint_mgr.load_parameters(
                from_file, network, strict=False, skip=skip,
                map_location=None)

    def init(self, args):
        self.args = args
        self.start_epoch = 0
        self.checkpoint_mgr = CheckpointManager(output=".", input=".")

    def make_optimizer(self, parameters):
        return optim.SGD(parameters, lr=0.0001, momentum=0.9)
        # return optim.Adam( parameters, lr=.01 )

    def gated_cpm(self):
        c3d_stages = [GatedStage("conv", 3, 1, 1, 1, 64, 4), GatedStage("pool", (1, 2, 2), (1, 2, 2), 0, 1, 0, 0),
                      GatedStage("conv", 3, 1, 1, 1, 128, 2), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                      GatedStage("conv", 3, 1, 1, 2, 256, 4), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                      GatedStage("conv", 3, 1, 1, 2, 512, 4), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                      GatedStage("conv", 3, 1, 1, 2, 512, 4), GatedStage("pool", 2, 2, 0, 1, 0, 0), ]

        fc_stages = [GatedStage("fc", 0, 0, 0, 2, 512, 2)]
        gate_modules = []

        for i, conv_stage in enumerate(c3d_stages):
            if conv_stage.name == "conv":
                for _ in range(conv_stage.nlayers):
                    count = strategy.PlusOneCount(strategy.UniformCount(conv_stage.ncomponents - 1))
                    gate_modules.append(strategy.NestedCountGate(conv_stage.ncomponents, count))
        for fc_stage in fc_stages:
            for _ in range(fc_stage.nlayers):
                count = strategy.PlusOneCount(strategy.UniformCount(fc_stage.ncomponents - 1))
                gate_modules.append(strategy.NestedCountGate(fc_stage.ncomponents, count))

        gate = strategy.SequentialGate(gate_modules)

        net = GatedC3D(gate, (21, 16, 45, 45), 5, c3d_stages, fc_stages)
        return net

##################################################################################################
# RL Learners
class PGLearner():

    def __init__(self, pgnet, data_network, train_dataset, reward, optimizer, to_device, device_ids=[0]):
        self.pgnet = pgnet
        self.loss = self.PGloss
        self.reward = reward
        self.train_dataset = train_dataset
        self.optimizer = optimizer
        self.to_device = to_device
        self.network = data_network
        self.device_ids = device_ids
        ngate_levels = 15
        inc = 1.0 / (ngate_levels - 1)
        self._us = [i * inc for i in range(ngate_levels)]

    def PGloss(self, y, reward):
        return -torch.log(y + .000001) * reward

    def eval_policy(self):
        pass

    # return EvaluationPolicy( self.env, self.dqn )
    def training_episode(self, rng, episode):
        self._train_batch(episode)

    def _train_batch(self, episode):
        for param_group in self.optimizer.param_groups:
            print("learning_rate: %s", param_group["lr"])

        # with training_mode( True, self ):
        self.optimizer.zero_grad()
        full_net = GestureNet()
        full_net.heatmap_net = self.to_device(full_net.heatmap_net)
        # cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i, data in enumerate(self.train_dataset):
            inputs, labels = data
            inputs = self.to_device(inputs)
            labels = self.to_device(labels)
            # generate intermidiate heatmaps
            heatmaps = full_net.get_heatmaps(inputs, torch.tensor(1.0))
            output = self.pgnet(heatmaps)
            exploration_rate = math.e ** (-0.01 * episode)
            randnum = np.random.uniform()
            if randnum < exploration_rate:
                a = np.random.randint(0, 14)
                print("RANDOM ACTION TAKEN")
            else:
                a = output.argmax(0).item()
            u = self._us[a]
            if u is None:  # Network turned off
                print("problem.step.yhat: None")
                yhat = None
                r = self.reward.reward(labels, yhat, x, torch.tensor([0.0]))
                gs = None
            else:
                u = self.to_device(torch.tensor([float(u)]))
                # u = self._tensor( u )
                x = self.to_device(heatmaps)
                u = self.to_device(u)
                yhat, gs = self.network(x, u)
                #print("problem.step.logits: %s", yhat)
                #print("problem.step.gs: %s", gs)
                p = fn.softmax(yhat, dim=1).squeeze()
                _, yhat = torch.max(p, dim=0)
                #print("problem.step.yhat: %s", yhat.item())
                r = self.reward.reward(labels, yhat.data, x, u)
            print("REWARD IS", r)
            loss = self.loss(output[a], r);
            print("LOSS", loss)
            loss.backward()
            self.optimizer.step()
            self._finish_batch()

    def _finish_batch(self):
        pass

#################################################################

class UsageAccuracyRewardModel:
    def __init__(self, data_network, all_flops):
        self.data_network = data_network
        self.all_flops = all_flops

    def reward(self, y, yhat, x, u):
        soft = torch.nn.Softmax(dim=1)
        print("--------------------------------------U-------------------------------------------", u)
        data_network_output = self.data_network(x, u)
        confidence_levels = soft(data_network_output[0])
        print("CONFIDENCE LEVELS:", confidence_levels * 100)
        decisions = data_network_output[1:]
        arrays = [decisions[0][i][0][0].cpu().numpy() for i in range(len(decisions[0]))]
        final_decisions = np.concatenate([array for array in arrays])
        list_flops = self.all_flops
        list_flops = [item for sublist in list_flops for item in sublist]
        list_flops = [i.macc for i in list_flops]
        flops_used = np.multiply(list_flops, final_decisions)
        print("LIST FLOPS: ", list_flops)
        print("FINAL DECISIONS: ", final_decisions)
        ratio_flops = np.sum(flops_used) / np.sum(list_flops)
        print("RATIO FLOPS", ratio_flops)
        if yhat.item() == y.item():
            print("NETWORK WAS CORRECT")
            print(confidence_levels[0].detach().cpu().numpy()[y.item()])
            return 2 * confidence_levels[0].detach().cpu().numpy()[y.item()] - ratio_flops  # return 50 * confidence_levels[0].numpy()[y.item()] -  ratio_flops
        else:
            print("NETWORK WAS INCORRECT")
            return -1 * ratio_flops

class DiscreteActionModel:
    def __init__(self, ngate_levels, include_none=True):
        assert ngate_levels >= 1
        if ngate_levels == 1:
            self._us = [1.0]
        else:
            inc = 1.0 / (ngate_levels - 1)
            self._us = [i * inc for i in range(ngate_levels)]
        # log.debug( "DiscreteActionModel.us: %s", self._us )
        self._include_none = include_none
        self._nactions = ngate_levels + (1 if include_none else 0)

    @property
    def action_space(self):
        return gym.spaces.Discrete(self._nactions)

    def action_to_control(self, a):
        assert isinstance(a, int)
        if a == len(self._us):
            assert self._include_none
            return None
        else:
            return self._us[a]

def initialize_weights(args):
    weight_init_fn = init.kaiming_normal_
    def impl(m):
        if model_util.is_weight_layer(m.__class__):
            print("init weights: %s", m)
            weight_init_fn(m.weight.data)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0)
    return impl

######################################################################################
class App(GatedNetworkApp):
    def __init__(self):
        #     parser = MyArgumentParser( description="RL control of gated networks",
        #       fromfile_prefix_chars="@", allow_abbrev=False )
        self.master_rng = random.Random(162)
        self.train_rng = random.Random()
        self.eval_rng = random.Random()

        def next_seed(seed=None):
            if seed is None:
                seed = self.master_rng.randrange(2 ** 31 - 1)
            random.seed(seed)
            self.train_rng.seed(seed + 10)
            self.eval_rng.seed(seed + 15)
            numpy.random.seed(seed + 20)
            torch.manual_seed(seed + 30)
            return seed

        seed = next_seed()
        #super().init( self.args )
        self.checkpoint_mgr = CheckpointManager(output=".", input=".")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.to_device = lambda t: t
        self.init_data_network()
        total, gated = self.data_network.flops((21, 16, 45, 45))
        self.all_flops = gated
        gtotal = sum(c.macc for m in gated for c in m)
        print(total)
        print(gtotal)
        print(total - gtotal)
        # self.init_dataset( self.args )
        self.start_epoch = 0  # Might get overwritten if loading checkpoint
        self.init_data()
        self.controller_macc = self.init_learner()

    def init_data(self):
        full_net = GestureNet()
        # full_net.heatmap_net.cuda()
        net = full_net.c3d_net
        from dataloaders.dataset import VideoDataset
        from torch.utils.data import DataLoader
        subset = ['No gesture', 'Thumb Down', 'Thumb Up', 'Swiping Left', 'Swiping Right']
        train_data = VideoDataset(dataset='20bn-jester', split='train', clip_len=16, subset=subset)
        batch_size = 1
        self.train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def init_learner(self):
        pgnet = ContextualBanditNet()
        pgnet.to(self.device)

        def make_explore():
            explore = "epsilon_greedy,constant,0.5"
            tokens = explore.split(",")
            if tokens[0] == "epsilon_greedy":
                spec = ",".join(tokens[1:])
                epsilon = schedule_spec("epsilon_greedy")(spec)
                # self.hyperparameters.append( epsilon )
                return MixturePolicy(
                    UniformRandomPolicy(self.train_env),
                    DqnPolicy(pgnet), epsilon)
            else:
                self.parser.error("--explore={} incompatible with DQN")

        #explore = make_explore()
        reward = self.make_reward_model()
        self.learner = PGLearner(pgnet, self.data_network, self.train_dataset, reward, self.make_optimizer(pgnet.parameters()),
                                 self.to_device)

        # First initialize parameters randomly because even when loading, the
        # feature network doesn't cover all of the parameters .
        self.init_network_parameters(pgnet, from_file=None)
        # self.init_network_parameters(features, from_file="/home/samyakp/Desktop/rl-solar-models/cifar10_resnet				# self.init_network_parameters(features, from_file="/home/samyakp/Desktop/rl-solar-models/cifar10_resnet8_model_150.pkl" )8_model_150.pkl" )
        #self.init_network_parameters(features, from_file="/home/samyak/Desktop/throttledemo/cpm_r3_model_epoch2000.pth")
        return util.flops(pgnet, (21,16, 45, 45)).macc

    def init_network_parameters(self, network, from_file=None):
        if from_file is not None:
            self.checkpoint_mgr.load_parameters(
                from_file, network, strict=False)
        else:
            # Initialize weights
            # FIXME: make `initialize_weights` a method of a superclass
            network.apply(initialize_weights("kaiming"))

    def init_data_network(self):
        self.data_network = self.gated_cpm()
        model_util.freeze(self.data_network)
        #### FIX THIS
        #self.data_network = FrozenBatchNorm(self.data_network)
        # Load or initialize parameters
        #         if (self.args.load_checkpoint is not None
        #             and self.args.load_data_network is not None):
        #           self.parser.error( "--load-checkpoint and --load-feature-network are"
        #                              " mutually exclusive" )
        # from_file = "/home/samyak/Desktop/rl-solar-models/cifar10_densenet_nested_model_310.pkl"#self.args.load_data_network
        from_file = "ckpt/cpm_r3_model_epoch1240.pth"

        #         if self.args.load_checkpoint is not None:
        #           from_file = self.checkpoint_mgr.get_checkpoint_file(
        #             "data_network", self.args.load_checkpoint )
        #           self.start_epoch = self.checkpoint_mgr.epoch_of_model_file( from_file )
        self.init_gated_network_parameters(self.data_network, from_file)
        self.data_network.to(self.device)

    def make_reward_model(self):
        return UsageAccuracyRewardModel(self.data_network, self.all_flops)

    def make_action_model(self):
        ngate_levels = 15
        return DiscreteActionModel(ngate_levels)

    def evaluate(self, policy, nepisodes, episode_length):
        print("evaluate: nepisodes: %s; episode_length: %s",
              nepisodes, episode_length)
        Vbar = MeanAccumulator()
        Tbar = MeanAccumulator()
        log = logging.getLogger(__name__)
        for ep in range(nepisodes):
            print("eval.%s.begin", ep)
            observers = [
                # rl.TrajectoryBuilder(),
                EpisodeLogger(log, logging.DEBUG, prefix="eval.{}.".format(ep)),
                SolarEpisodeLogger(log, logging.INFO)]
            (T, V) = episode(
                self.eval_rng, self.eval_env, policy,
                observer=EpisodeObserverList(*observers), time_limit=episode_length)

            # Vbar( V.squeeze()[0] )
            Vbar(V.item())
            Tbar(T)

            # Format is important for log parsing
            print("eval.%s.t: %s%s", ep, T, " *" if T == episode_length else "")
            print("eval.%s.v: %s", ep, V.item())
        return Tbar.mean(), Vbar.mean()

    def checkpoint(self, elapsed_episodes, force_eval=False):
        #         milestone = (force_eval
        #                      or (elapsed_episodes % self.args.checkpoint_interval == 0))
        milestone = True

        def save_fn(name, network):
            self.checkpointkpoint_mgr.save_checkpoint(
                name, network, elapsed_episodes,
                data_parallel=self.args.data_parallel, persist=milestone)

        # self.learner.apply_to_modules( save_fn )
        # save_fn( "data_network", self.data_network )

        if milestone:
            # Format is important for log parsing
            print("* Episode %s", elapsed_episodes)

            eval_policy = self.learner.eval_policy()
            eval_episodes = 3  # 5
            eval_episode_length = 5  # 1000
            tmean, vmean = self.evaluate(eval_policy,
                                         eval_episodes, eval_episode_length)
            print("* eval.vmean: %s", vmean)
            print("* eval.tmean: %s", tmean)

    #         if self.args.post_checkpoint_hook is not None:
    #           os.system( self.args.post_checkpoint_hook )
    def main(self):
        def set_epoch(epoch_idx, nbatches):
            print("training: epoch: %s; nbatches: %s", epoch_idx, nbatches)

        #           for hp in self.hyperparameters:
        #             hp.set_epoch( epoch_idx, nbatches )
        #             print( hp )
        def set_batch(batch_idx):
            print("batch: %s", batch_idx)
            for hp in self.hyperparameters:
                hp.set_batch(batch_idx)
                print(hp)

        print("==================== Start ====================")
        start = self.start_epoch
        print("start: epoch: %s", start)
        # Save initial model if not resuming
        # if self.args.load_checkpoint is None:
        # self.checkpoint( 0 )
        # Training loops
        train_episodes = 1
        for ep in range(start, start + train_episodes):
            print("EPISODE NUMBER: ", ep)
            set_epoch(ep, nbatches=1)
            # Update learning rate
            for param_group in self.learner.optimizer.param_groups:
                break
                param_group["lr"] = self.args.learning_rate()
            self.learner.training_episode(
                self.train_rng, ep)

        # self.checkpoint( ep+1 )
    # Save final model if we haven't done so already
    # checkpoint_interval = 2

    # if train_episodes % checkpoint_interval != 0:
    #   self.checkpoint( train_episodes, force_eval=True )


class CheckpointManager:
    def __init__(self, *, output, input=None):
        self.output = output
        self.input = input

    def model_file(self, directory, prefix, epoch, suffix=""):
        filename = "{}_{}.pkl{}".format(prefix, epoch, suffix)
        return os.path.join(directory, filename)

    def latest_checkpoints(self, directory, name):
        return glob.glob(os.path.join(directory, "{}_*.pkl.latest".format(name)))

    def epoch_of_model_file(self, path):
        m = re.match(".*_([0-9]+)\\.pkl(\\.latest)?", os.path.basename(path)).group(1)
        return int(m)

    def load_parameters(self, path, network, strict=True, skip=None, map_location=None):
        if skip is None:
            skip = lambda param_name: False

        with open(path, "rb") as fin:
            state_dict = torch.load(fin, map_location="cpu")

        own_state = network.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                # log.verbose( "Load %s", name )
                if skip(name):
                    # log.verbose( "Skipping module" )
                    continue
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError("unexpected key '{}' in state_dict".format(name))
            else:
                # log.warning( "unexpected key '{}' in state_dict".format(name) )
                pass
        missing = set(own_state.keys()) - set(state_dict.keys())
        missing = [k for k in missing if not skip(k)]
        if len(missing) > 0:
            if strict:
                raise KeyError("missing keys in state_dict: {}".format(missing))
            else:
                #print( "missing keys in state_dict: {}".format(missing) )
                pass
###########################################################################
app = App()
app.main()