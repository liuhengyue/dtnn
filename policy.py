import abc
class Policy(metaclass=abc.ABCMeta):
    def reset(self):
        pass

    @abc.abstractmethod
    def observe(self, s):
        raise NotImplementedError()

    def __call__(self, rng, s):
        return self.observe(s)(rng)


class MixturePolicy(Policy):
    def __init__(self, pi_less, pi_greater, epsilon):
        """
        Parameters:
            `pi_less` : Execute with probability `epsilon`
            `pi_greater` : Execute with probability `1 - epsilon`
            `epsilon` : A `Hyperparameter` giving the schedule for epsilon.
        """
        super().__init__()
        self.pi_less = pi_less
        self.pi_greater = pi_greater
        self.epsilon = epsilon
        self._nsteps = 0

    def reset(self):
        # Note that we don't reset the step count
        print("MixturePolicy.epsilon: %s", self.epsilon())
        self.pi_less.reset()
        self.pi_greater.reset()

    def observe(self, s):
        # FIXME: Should compute these lazily in action() and cache them
        pi_less_action = self.pi_less.observe(s)
        pi_greater_action = self.pi_greater.observe(s)
        self.epsilon.set_epoch(self._nsteps, nbatches=1)
        epsilon = self.epsilon()  # Capture current value
        self._nsteps += 1

        def action(rng):
            if epsilon > 0 and rng.random() < epsilon:
                return pi_less_action(rng)
            else:
                return pi_greater_action(rng)

        return action


class DqnPolicy(Policy):
    def __init__(self, dqn):
        super().__init__()
        self.dqn = dqn

    def observe(self, s):
        def action(rng):
            if not hasattr(action, "cached"):
                q = self.dqn(s)
                assert (len(q.size()) == 2 and q.size(0) == 1)
                action.cached = torch.argmax(q, dim=1).item()
            return action.cached

        return action


class EvaluationPolicy(DqnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class UniformRandomPolicy(Policy):
    def __init__(self, env):
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                "Don't know how to sample from {}".format(env.action_space))
        self.action_space = env.action_space

    def observe(self, s):
        def action(rng):
            # TODO: Generalize to other spaces
            return rng.randrange(self.action_space.n)

        return action