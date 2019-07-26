"""
Gambler's problem MDPd for exercises 4.9
"""


from typing import Tuple
from itertools import product

from toolz import curry

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from distribution import *
from mdp import *


@curry
def clamp(smallest, largest, n):
    return max(smallest, min(n, largest))


# Action is number of cars to transfer from first to second
Action = int    # strictly from range(1, 100)

actions = list(range(1, 99))

# State is remaining capital
State = int

states = range(101)

clip = clamp(0, 100)


def terminality(s: State) -> bool:
    return (s == 0) or (s == 100)


class OutcomeDistribution(Distribution[Tuple[float, State]]):
    def __init__(self, state: State, action: Action, p: float):
        # available to rent is end of day + impact of moved vehicles
        self.capital = state
        self.bet = action
        self.total = None
        self.prob_fn = lambda: self._outcomes(p)

    def _outcomes(self, p: float):
        return Distribution(lambda: [
            Weighted((1. if self.bet + self.capital >= 100 else 0., clip(self.capital + self.bet)), p),
            Weighted((0., clip(self.capital - self.bet)), 1. - p)
        ])


def transitions(p: float):
    result = {}
    for s in states:
        for action in actions:
            # trim out illegal actions
            if (action <= s) and (action <= 100 - s):
                result[(s, action)] = StrictDistribution(lambda: OutcomeDistribution(s,
                                                                                     action,
                                                                                     p))
    return result


def print_v(V: Dict[State, float]):
    for s, v in V.items():
        print("\t{}: {}".format(s, round(v, 4)))


def print_q(Q: Dict[Tuple[State, Action], float]):
    for (s, a), v in Q.items():
        print("\tFrom {} {}: {}".format(s, a, round(v, 4)))


def print_policy(p: Policy):
    for s in states:
        print("\tIn {} select from {}".format(s, p(s)))


def plot_policy(policy: Policy, title: str, name: str):
    policy_actions = np.zeros(100)
    for idx in range(len(policy_actions)):
        policy_actions[idx] = policy(idx+1).sample()

    fig, ax = plt.subplots()
    plt.bar(range(1, 101), policy_actions)
    ax.set_title(title)
    plt.savefig(name + ".png")
    plt.close()


if __name__ == "__main__":
    mdp = MDP(System(states,
                     actions,
                     transitions(0.4),
                     terminality=terminality), (0, 0), 1.0)

    # q = mdp.optimize_q(0.00001)
    # print_q(q)
    policy = MemoizedPolicy(lambda _: Distribution.deterministic(1))
    p_star, v = mdp.optimize_policy(policy,
                                    eval_threshold=1e-6,
                                    use_value_update=True,
                                    greedy_prob=0.5)
    print("pi* for (4.9) p=0.4:")
    print_policy(p_star)
    print_v(v)
    plot_policy(p_star, "Gambler dynamics for p=0.4", "gambler_dynamics_04")

    mdp = MDP(System(states,
                     actions,
                     transitions(0.25),
                     terminality=terminality), (0, 0), 1.0)
    p_star, v = mdp.optimize_policy(policy,
                                    eval_threshold=1e-6,
                                    use_value_update=True,
                                    greedy_prob=0.5,
                                    action_stability_margin=0.00001)
    print("pi* for (4.9) p=0.25:")
    print_policy(p_star)
    print_v(v)
    plot_policy(p_star, "Gambler dynamics for p=0.25", "gambler_dynamics_025")

    mdp = MDP(System(states,
                     actions,
                     transitions(0.55),
                     terminality=terminality), (0, 0), 1.0)
    p_star, v = mdp.optimize_policy(policy,
                                    eval_threshold=1e-6,
                                    use_value_update=True,
                                    greedy_prob=0.5)
    print("pi* for (4.9) p=0.55:")
    print_policy(p_star)
    print_v(v)
    plot_policy(p_star, "Gambler dynamics for p=0.4", "gambler_dynamics_055")
