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

import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init

import gym
import gym.spaces

from network.gated_c3d import GatedC3D
from network.gated_c3d import GatedStage
from network.gated_c3d import C3dDataNetwork
from nnsearch.pytorch.gated import strategy
from nnsearch.pytorch.parameter import *
from nnsearch.pytorch.rl import *
# todo: eval function has some logger stuff
from nnsearch.pytorch.rl.env.solar import SolarEpisodeLogger
from nnsearch.pytorch.rl.policy import *
from nnsearch.pytorch.rl.dqn import *
from nnsearch.statistics import *
from nnsearch.pytorch.checkpoint import CheckpointManager
from nnsearch.pytorch.gated.module import BlockGatedConv3d, BlockGatedConv2d, BlockGatedFullyConnected
from network.demo_model import GestureNet

from bandit_net import ContextualBanditNet
import shape_flop_util as util
import model_util

from modules.utils import latest_checkpoints
from tqdm import tqdm
from datetime import datetime
from collections import Counter

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
        # return optim.SGD(parameters, lr=1e-5, momentum=0.9)
        # return optim.Adam( parameters, lr=1e-2 )
        return optim.RMSprop(parameters, lr=1e-5)

    def gated_network(self):
        return C3dDataNetwork(num_classes=27)

##################################################################################################
# RL Learners
class PGLearner():

    def __init__(self, pgnet, data_network, train_dataset, test_dataset, reward, optimizer, to_device, device_ids=[1], batch_size=None):
        self.pgnet = pgnet
        self.loss = self.PGloss
        self.reward = reward
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.to_device = to_device
        self.network = data_network
        self.device_ids = device_ids
        self.batch_size = batch_size
        self.ngate_levels = 10
        self.inc = 1.0 / self.ngate_levels
        # u = 0 does not make sense
        # if 10 levels: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
        self._us = torch.tensor([i * self.inc for i in range(1, self.ngate_levels+1)], requires_grad=False).to(self.device_ids[0])
        # (10, B) [[0.1, 0.1, ...], [0.2, 0.2, ...], ..., [1.0, 1.0, ...]]
        self.u_space = torch.repeat_interleave(self._us, self.batch_size, dim=0).view(-1, self.batch_size)
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        # print(self._us)

    def PGloss(self, yhat, reward, action):
        # return torch.mean(-torch.log(yhat + .000001) * reward)
        # return F.smooth_l1_loss(yhat, reward)
        # https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
        # print(reward)
        # print(torch.log(yhat))
        # add weights to contextual net prediction
        # (B, 10)
        reward[torch.arange(0, self.batch_size, dtype=torch.long), action] *= 2.0
        # l2 normalize rewards
        reward /= torch.norm(reward, dim=1).view(-1, 1)

        return torch.mean(-torch.log(yhat + torch.finfo().eps) * reward)

    def eval_policy(self):
        self.pgnet.eval()

        nclasses = len(self.test_dataset.dataset.class_names)
        batch_size = self.test_dataset.batch_size
        class_correct = [0.0] * nclasses
        class_total = [0.0] * nclasses
        u_history = 0.0
        u_bins = {}
        for i in range(0, self.ngate_levels):
            u_bins[i] = 0
        with torch.no_grad():
            for _, data in enumerate(tqdm(self.test_dataset)):
                inputs, labels = data
                inputs = inputs.to(self.device_ids[0])
                labels = labels.to(self.device_ids[0])
                output = self.pgnet(inputs)
                a = torch.argmax(output, 1)
                u = torch.take(self._us, a)
                for k in a.cpu().numpy():
                    u_bins[k] += 1
                u_history += torch.sum(u).item()
                # print("---- u ----", u)
                yhat, gs = self.network(inputs, u)
                _, predicted = torch.max(yhat.data, 1)
                # print("---- pred ----", predicted)
                c = (predicted == labels).cpu().numpy()
                for i in range(len(c)):
                    label = labels[i]
                    class_correct[label] += c[i]
                    class_total[label] += 1
            log.info("controller network test, total %s [%s/%s]", sum(class_correct) / sum(class_total), sum(class_correct), sum(class_total))
            log.info("Average u: %s", u_history / sum(class_total))
            log.info("Running u bins: %s", u_bins)
            for i in range(nclasses):
                if class_total[i] > 0:
                    log.info("'%s' : %s [%s/%s]", self.test_dataset.dataset.class_names[i],
                             class_correct[i] / class_total[i], class_correct[i], class_total[i])
                else:
                    log.info("'%s' : None", self.test_dataset.dataset.class_names[i])

    # return EvaluationPolicy( self.env, self.dqn )
    def training_episode(self, rng, episode):
        self._train_batch(episode)


    def _train_batch(self, episode):
        for param_group in self.optimizer.param_groups:
            print("learning_rate: %s", param_group["lr"])

        # with training_mode( True, self ):

        running_corrects = 0.0
        running_loss = 0.0
        running_reward = 0.0
        u_bins = {}
        for i in range(0, self.ngate_levels):
            u_bins[i] = 0
        u_history = 0.0
        exploration_rate = math.e ** (-0.5 * (episode + 1))
        # exploration_rate = 0
        print("Exploration rate: %s", exploration_rate)
        log.info("Exploration rate: %s", exploration_rate)
        # cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i, data in enumerate(tqdm(self.train_dataset)):
            self.optimizer.zero_grad()
            inputs, labels, _ = data
            inputs = inputs.to(self.device_ids[0])
            labels = labels.to(self.device_ids[0])
            # generate psudo gt for RL, no gesture 0 [u = 0.1], others 9 [u = 1.0]
            # u_gts = torch.where(labels == 0, torch.zeros(labels.size(), device=labels.device), 9 * torch.ones(labels.size(), device=labels.device)).long()
            # size (B, 5 * 10)
            output = self.pgnet(inputs)
            # print("# **** output ****\n", output.detach().cpu().numpy())

            # exploration_rate = 0.0
            randnum = np.random.uniform()
            # print(randnum, exploration_rate)
            if randnum < exploration_rate:
                action = torch.randint(0, self.ngate_levels, (self.batch_size,), device=output.device)
                # print("RANDOM ACTION TAKEN")
            else:
                # a = output.argmax(0).item()
                action = torch.argmax(output, 1)
                # print(torch.max(output, 1)[0])

            # a ranges from 0 to 9

            # print("#1 STATE -----------\n", state.detach().cpu().numpy())

            u = torch.take(self._us, action)
            # print("u ------\n", u.detach().cpu().numpy())
            for k in action.cpu().numpy():
                u_bins[k] += 1

            u_history += torch.sum(u).item()
            # print("# ------ u ------", u)

            # loop over each possible u in [0.1,...,1.0] for each example
            rewards = []
            for m in range(self.ngate_levels):
                yhat, gs = self.network(inputs, self.u_space[m])
                # yhats.append(yhat)
            # size (B, 27 * 10)
            # yhats = torch.cat(yhats, dim=1)
            # print(yhats.size())
            #print("problem.step.logits: %s", yhat)
            #print("problem.step.gs: %s", gs)

            #print("problem.step.yhat: %s", yhat.item())
                r = self.reward.reward(labels, yhat, gs)
                rewards.append(r)

            # rewards matrix for all u
            rewards = torch.stack(rewards).t_()

            # if randnum > exploration_rate:
            #     print("#2 REWARD -----------\n", r.detach().cpu().numpy())
            # print("#3 PGNET OOUTPUT -----------\n", output.detach().cpu().numpy())
            # take STATE as ground-truth
            # print(output)
            # print(state)
            # print(r)
            # logits = self.ce(output,  u_gts)
            state_probs = torch.nn.Softmax(dim=1)(output)
            # print(state_probs)
            # picked_state_probs = torch.gather(state_probs, 1, state.view(-1, 1)).view(-1)

            # ce_loss = self.ce(logits, u_gts)
            # print(ce_loss)
            # print("# State softmax -----------\n", state_probs.detach().cpu().numpy())
            # print("# Picked state softmax -----------\n", picked_state_probs.detach().cpu().numpy())
            # print("# Reward -----------\n", r.detach().cpu().numpy())
            # print("logits: ", logits)
            # put more weights on picked u
            loss = self.loss(state_probs, rewards, action)
            # print("# Loss -----------\n", loss.detach().cpu().numpy())
            # probs = torch.nn.Softmax(dim=1)(yhat)
            # preds = torch.max(probs, 1)[1]
            # batch_corrects = torch.sum(preds == labels.data).item()
            # running_corrects += batch_corrects
            running_loss += loss.item()
            predict_rewards = rewards[torch.arange(0, self.batch_size, dtype=torch.long), action]
            running_reward += torch.mean(predict_rewards).item()

            if i % 10 == 0:
                running_num = (i + 1) * self.train_dataset.batch_size
                print("Running loss: ", running_loss / running_num)
                print("Running reward: ", running_reward / running_num)
                # print("Running corrects: ", running_corrects / running_num)
                print("Running average u: ", u_history / running_num)
                print("Running u bins: ", u_bins)
            if i % 400 == 0:
                running_num = (i + 1) * self.train_dataset.batch_size
                log.info("Step - %s", i)
                log.info("Running loss: %s", running_loss / running_num)
                log.info("Running reward: %s", running_reward / running_num)
                # log.info("Running corrects: %s", running_corrects / running_num)
                log.info("Running average u: %s ", u_history / running_num)
                log.info("Running u bins: %s", u_bins)

            # if i == 100:
            #     break


            # print("LOSS", loss)
            loss.backward()
            self.optimizer.step()
            self.pgnet.zero_grad()
        self._finish_batch()

        # end of epoch
        running_num = len(self.train_dataset) * self.train_dataset.batch_size
        log.info("End of epoch %s", episode)
        log.info("Loss: %s", running_loss / running_num)
        log.info("Reward: %s", running_reward / running_num)
        log.info("Accuracy: %s", running_corrects / running_num)

    def _finish_batch(self):
        pass

#################################################################

class UsageAccuracyRewardModel:
    def __init__(self, data_network, all_flops, device_ids):
        self.device_ids = device_ids
        self.data_network = data_network
        self.F = self.total_flops(all_flops)
        self.total_F = torch.sum(self.F)
        # self.ce = torch.nn.MSELoss(reduction='none')
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
    def total_flops(self, all_flops):
        list_flops = [item for sublist in all_flops for item in sublist]
        list_flops = [i.macc for i in list_flops]
        return torch.tensor(list_flops, device=self.device_ids[0], requires_grad=False)

    def reward(self, y, yhat, gs):
        '''

        :param y: (B)
        :param state_class: (B, 1)
        :param yhat: (B, 5)
        :param gs: list of G
        :return: Reward (B,)
        '''
        yhat = yhat.detach()
        with torch.no_grad():
            # gs is a list of gate matrix, each matrix is of (B, ncomponents)
            # print("yhat: ", yhat)
            # (B, 27)
            confidence_levels = torch.nn.Softmax(dim=1)(yhat)

            # print("CONFIDENCE LEVELS:", confidence_levels * 100)
            # print("gs *********", [g[0].detach() for g in gs])
            # concatenated gate matrices of (B, 112(may vary))
            G = torch.cat([g[0] for g in gs], 1)
            # print("Test on gs: @@@@@@@@@@@@@@@@@", G.size(), G.requires_grad)
            # arrays = [gs[i][0][0].cpu().numpy() for i in range(len(gs))]
            # final_decisions = np.concatenate([array for array in arrays])
            # flops used for each input of a batch (B, )
            flops_used = torch.sum(G * self.F, 1)


            ratio_flops = flops_used / self.total_F
            # print("RATIO FLOPS ------\n", ratio_flops.detach().cpu().numpy())

            pred_max, pred = torch.max(confidence_levels, dim=1)

            batch_pred_bool = pred == y
            # print("before if: ", pred, y, pred.requires_grad)
            # print(y.view(-1, 1))
            # (B, 1) -> (B,) should be a better way
            # gt_confidence_levels = torch.gather(confidence_levels, 1, y.view(-1, 1)).view(-1)
            # positive_r = torch.max(gt_confidence_levels - ratio_flops,
            #                        torch.zeros(gt_confidence_levels.size(), device=gt_confidence_levels.device))
            # positive_r = torch.exp(gt_confidence_levels - ratio_flops)
            # different reward for 'No gesture' class
            # ratio_flops = torch.where(y == 0, ratio_flops / 5, ratio_flops / 2)
            # approximately -1 to 1

            # put class weights on flops
            # print(yhat)
            # print(confidence_levels)
            # batch_acc = torch.mean(batch_pred_bool.float()).detach().cpu().numpy()

            # print("#2 PREDICTION ACCURACY -----------\n", batch_acc)
            # print(pred_diff * ratio_flops)
            # print(ratio_flops)

            # positive_r = ((1.5 - ratio_flops) * (0.1 + pred_max)) ** 2
            # positive_r = pred_max - ratio_flops + 0.5
            positive_r = 1 - ratio_flops
            # positive_r = ratio_flops
            # negative_r less -> ratio_flops more -> pred_diff less
            # negative_r = -1 / torch.exp( - ratio_flops * pred_max)
            # negative_r = torch.ones(positive_r.size(), device=positive_r.device) * (-1)
            negative_r = - (pred_max + 0.5) * (ratio_flops + 1.5)

            r = torch.where(batch_pred_bool, positive_r, negative_r)
            # l2 normalize
            # r_norm = torch.norm(r)
            # r = r / r_norm

            # r = (r - torch.mean(r)) / (torch.std(r) + 1e-16)

            return r

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
    def __init__(self, start_epoch=0, device_ids=["cpu"], batch_size_per_gpu=4, mode="train"):
        #     parser = MyArgumentParser( description="RL control of gated networks",
        #       fromfile_prefix_chars="@", allow_abbrev=False )
        # self.master_rng = random.Random(162)
        # self.train_rng = random.Random()
        # self.eval_rng = random.Random()
        #
        # def next_seed(seed=None):
        #     if seed is None:
        #         seed = self.master_rng.randrange(2 ** 31 - 1)
        #     random.seed(seed)
        #     self.train_rng.seed(seed + 10)
        #     self.eval_rng.seed(seed + 15)
        #     numpy.random.seed(seed + 20)
        #     torch.manual_seed(seed + 30)
        #     return seed
        #
        # seed = next_seed()
        #super().init( self.args )
        self.checkpoint_mgr = CheckpointManager(output="ckpt/controller", input=".")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ids = device_ids
        self.data_parallel = True if len(device_ids) > 1 else False
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size = self.batch_size_per_gpu * len(self.device_ids)
        # self.device = "cpu"
        self.to_device = lambda t: t
        self.mode = mode
        total, gated = self.init_data_network()
        self.all_flops = gated
        gtotal = sum(c.macc for m in gated for c in m)
        # print(total)
        print("Total gated modules flops - {}".format(gtotal))
        # print(total - gtotal)
        # self.init_dataset( self.args )
        self.start_epoch = start_epoch
        self.init_data()
        self.controller_macc = self.init_learner()

    def init_data(self):

        from dataloaders.dataset import VideoDataset
        from torch.utils.data import DataLoader
        subset = ['No gesture', 'Swiping Down', 'Swiping Left', 'Swiping Right', 'Swiping Up']
        subset = None
        train_data = VideoDataset(dataset='20bn-jester', split='train', clip_len=16, subset=subset)
        self.train_dataset = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_data = VideoDataset(dataset='20bn-jester', split='val', clip_len=16, subset=subset)
        self.test_dataset = DataLoader(test_data, batch_size=2*self.batch_size, shuffle=False, drop_last=False)

    def init_learner(self):
        pgnet = ContextualBanditNet()
        pgnet.to(self.device_ids[0])

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


        # First initialize parameters randomly because even when loading, the
        # feature network doesn't cover all of the parameters .
        if self.start_epoch > 0 or self.mode == "test":
            saved_controller_file = self.checkpoint_mgr.latest_checkpoints("ckpt/controller/", "controller_network")[0]
        else:
            saved_controller_file = None

        self.init_network_parameters(pgnet, from_file=saved_controller_file)

        # self.init_network_parameters(features, from_file="/home/samyakp/Desktop/rl-solar-models/cifar10_resnet				# self.init_network_parameters(features, from_file="/home/samyakp/Desktop/rl-solar-models/cifar10_resnet8_model_150.pkl" )8_model_150.pkl" )
        #self.init_network_parameters(features, from_file="/home/samyak/Desktop/throttledemo/cpm_r3_model_epoch2000.pth")
        flops = util.flops(pgnet, (3, 16, 100, 160)).macc
        print("Controller network flops - {}".format(flops))
        if len(self.device_ids) > 1:
            pgnet = torch.nn.DataParallel(pgnet, device_ids=self.device_ids)
            print("Policy network - using multi-gpus: ", self.device_ids)

        self.learner = PGLearner(pgnet, self.data_network, self.train_dataset, self.test_dataset,
                                 reward, self.make_optimizer(pgnet.parameters()), self.to_device, self.device_ids, batch_size=self.batch_size)

        return flops

    def init_network_parameters(self, network, from_file=None):
        if from_file is not None:
            self.checkpoint_mgr.load_parameters(
                from_file, network, strict=False)
        else:
            # Initialize weights
            # FIXME: make `initialize_weights` a method of a superclass
            network.apply(initialize_weights("kaiming"))

    def init_data_network(self):
        self.data_network = self.gated_network()
        model_util.freeze(self.data_network)
        #### FIX THIS
        self.data_network = model_util.FrozenBatchNorm(self.data_network)
        # Load or initialize parameters
        #         if (self.args.load_checkpoint is not None
        #             and self.args.load_data_network is not None):
        #           self.parser.error( "--load-checkpoint and --load-feature-network are"
        #                              " mutually exclusive" )
        # from_file = "/home/henry/Research/throttling-demo/ckpt/gated_raw_c3d/model_95.pkl.good"
        from_file = self.checkpoint_mgr.latest_checkpoints("ckpt/gated_raw_c3d/", "model")[0]

        #         if self.args.load_checkpoint is not None:
        #           from_file = self.checkpoint_mgr.get_checkpoint_file(
        #             "data_network", self.args.load_checkpoint )
        #           self.start_epoch = self.checkpoint_mgr.epoch_of_model_file( from_file )
        self.init_gated_network_parameters(self.data_network, from_file)
        total, gated = self.data_network.flops((3, 16, 100, 160))
        self.data_network.to(self.device_ids[0])
        if len(self.device_ids) > 1:
            self.data_network = torch.nn.DataParallel(self.data_network, device_ids=self.device_ids)
            print("Data network - using multi-gpus: ", self.device_ids)

        return total, gated

    def make_reward_model(self):
        return UsageAccuracyRewardModel(self.data_network, self.all_flops, self.device_ids)

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
        checkpoint_interval = 1
        milestone = (force_eval and (elapsed_episodes % checkpoint_interval == 0))

        # self.learner.apply_to_modules( save_fn )
        if elapsed_episodes % checkpoint_interval == 0:
            self.checkpoint_mgr.save_checkpoint(
                "controller_network", self.learner.pgnet, elapsed_episodes,
                data_parallel=self.data_parallel, persist=True)

        # if milestone:
        #     # Format is important for log parsing
        #     print("* Episode %s", elapsed_episodes)
        #
        #     eval_policy = self.learner.eval_policy()
        #     eval_episodes = 3  # 5
        #     eval_episode_length = 5  # 1000
        #     tmean, vmean = self.evaluate(eval_policy,
        #                                  eval_episodes, eval_episode_length)
        #     print("* eval.vmean: %s", vmean)
        #     print("* eval.tmean: %s", tmean)

    #         if self.args.post_checkpoint_hook is not None:
    #           os.system( self.args.post_checkpoint_hook )
    def main(self):
        def set_epoch(epoch_idx, nbatches):
            print("training: epoch: %s; nbatches: %s", epoch_idx, nbatches)
            log.info("training: epoch: %s", epoch_idx)

        #           for hp in self.hyperparameters:
        #             hp.set_epoch( epoch_idx, nbatches )
        #             print( hp )
        def set_batch(batch_idx):
            print("batch: %s", batch_idx)
            for hp in self.hyperparameters:
                hp.set_batch(batch_idx)
                print(hp)
        if self.mode == "train":
            # check if data network is frozen
            # for p in self.data_network.parameters():
            #     print(p.requires_grad)
            print("==================== Start ====================")
            start = self.start_epoch
            print("start: epoch: %s", start)
            # Save initial model if not resuming
            # if self.args.load_checkpoint is None:
            # self.checkpoint( 0 )
            # Training loops
            train_episodes = 10
            for ep in range(start, start + train_episodes):
                print("EPISODE NUMBER: ", ep)
                set_epoch(ep, nbatches=1)
                # Update learning rate
                # for param_group in self.learner.optimizer.param_groups:
                #     param_group["lr"] = self.args.learning_rate()
                self.learner.training_episode(
                    None, ep)
                # self.learner.training_episode(
                #     self.train_rng, ep)

                self.checkpoint( ep+1 )
                # eval after each epoch
                # self.learner.eval_policy()
        elif self.mode == "test":
            self.learner.eval_policy()

    # Save final model if we haven't done so already
    # checkpoint_interval = 2

    # if train_episodes % checkpoint_interval != 0:
    #   self.checkpoint( train_episodes, force_eval=True )


###########################################################################
if __name__ == "__main__":
    # Logger setup
    # mylog.add_log_level("VERBOSE", logging.INFO - 5)
    # mylog.add_log_level("MICRO", logging.DEBUG - 5)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Need to set encoding or Windows will choke on ellipsis character in
    # PyTorch tensor formatting
    mode = "train"
    experiment_name = 'throttle_demo_' + mode
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join("logs", experiment_name + '_' + timestamp + '.log')
    handler = logging.FileHandler(log_path, "w", "utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    root_logger.addHandler(handler)
    app = App(start_epoch=0, device_ids=[0], batch_size_per_gpu=5, mode=mode)
    app.main()
