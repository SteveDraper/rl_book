"""
Gambler's problem MDPd for exercises 4.9
"""


from typing import Tuple
from itertools import product

from toolz import curry

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

from distribution import *
from mdp import *


@curry
def clamp(smallest, largest, n):
    return max(smallest, min(n, largest))


# Action is number of cars to transfer from first to second
Action = int    # strictly from range(1, 100)

WIN_SCORE = 100

actions = list(range(1, WIN_SCORE))

# State is remaining capital
State = int

states = range(WIN_SCORE+1)  # states 0 and WIN_SCORE are terminal

clip = clamp(0, WIN_SCORE)


def terminality(s: State) -> bool:
    return (s == 0) or (s == WIN_SCORE)


class OutcomeDistribution(Distribution[Tuple[float, State]]):
    def __init__(self, state: State, action: Action, p: float):
        # available to rent is end of day + impact of moved vehicles
        self.capital = state
        self.bet = action
        self.total = None
        self.prob_fn = lambda: self._outcomes(p)

    def _outcomes(self, p: float):
        return Distribution(lambda: [
            Weighted((1. if self.bet + self.capital >= WIN_SCORE else 0., clip(self.capital + self.bet)), p),
            Weighted((0., clip(self.capital - self.bet)), 1. - p)
        ])


def transitions(p: float):
    result = {}
    for s in states:
        for action in actions:
            # trim out illegal actions
            if (action <= s) and (action <= WIN_SCORE - s):
                result[(s, action)] = StrictDistribution(lambda: OutcomeDistribution(s,
                                                                                     action,
                                                                                     p))
    return result


def print_v(V: Dict[State, float]):
    for s, v in V.items():
        print("\t{}: {}".format(s, round(v, 7)))


def print_q(Q: Dict[Tuple[State, Action], float]):
    for (s, a), v in Q.items():
        print("\tFrom {} {}: {}".format(s, a, round(v, 7)))


def print_policy(p: Policy):
    for s in states:
        print("\tIn {} select from {}".format(s, p(s)))


def plot_policy(policy: Policy, title: str, name: str):
    policy_actions = np.zeros(WIN_SCORE-1)
    for idx in range(len(policy_actions)):
        policy_actions[idx] = policy(idx+1).sample()

    fig, ax = plt.subplots()
    plt.bar(range(1, WIN_SCORE), policy_actions)
    ax.set_title(title)
    plt.savefig(name + ".png")
    plt.close()


def plot_state_q(q: Dict[Tuple[State, Action], float],
                 title: str,
                 name: str,
                 s: State=0):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    action_vals = np.zeros((WIN_SCORE-1, WIN_SCORE-1))
    for (st, a), v in q.items():
        if (st > s) and (st < WIN_SCORE):
            action_vals[st-1, a-1] = v

    X, Y = np.meshgrid(np.arange(1, WIN_SCORE), np.arange(WIN_SCORE-1, 0, -1))
    x = np.ravel(X)
    y = np.ravel(Y)
    z = np.zeros(len(x))
    for idx in range(len(x)):
        z[idx] = action_vals[y[idx]-1, x[idx]-1]
    Z = z.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)

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
                                    eval_threshold=1e-8,
                                    use_value_update=True,
                                    greedy_prob=0.5,
                                    action_stability_margin=1e-6)
    # print("pi* for (4.9) p=0.4:")
    # print_policy(p_star)
    # print_v(v)
    plot_policy(p_star, "Gambler dynamics for p=0.4", "gambler_dynamics_04")

    q_star = mdp.optimize_q(1e-8)
    # print_q(q_star)
    plot_state_q(q_star, "Gambler q distribution for state > 48", "gambler_q49", s=WIN_SCORE/2-1)

    mdp = MDP(System(states,
                     actions,
                     transitions(0.25),
                     terminality=terminality), (0, 0), 1.0)
    p_star, v = mdp.optimize_policy(policy,
                                    eval_threshold=1e-6,
                                    use_value_update=True,
                                    greedy_prob=0.5,
                                    action_stability_margin=0.00001)
    # print("pi* for (4.9) p=0.25:")
    # print_policy(p_star)
    # print_v(v)
    plot_policy(p_star, "Gambler dynamics for p=0.25", "gambler_dynamics_025")

    mdp = MDP(System(states,
                     actions,
                     transitions(0.55),
                     terminality=terminality), (0, 0), 1.0)
    p_star, v = mdp.optimize_policy(policy,
                                    eval_threshold=1e-6,
                                    use_value_update=True,
                                    greedy_prob=0.5)
    # print("pi* for (4.9) p=0.55:")
    # print_policy(p_star)
    # print_v(v)
    plot_policy(p_star, "Gambler dynamics for p=0.4", "gambler_dynamics_055")

    # print_q(q_star)
    sub_mid_score = WIN_SCORE/2-1
    print("For p=0.4: q({},1) = {}, q({},{}) = {}".format(sub_mid_score,
                                                          q_star[(sub_mid_score, 1)],
                                                          sub_mid_score,
                                                          sub_mid_score,
                                                          q_star[(sub_mid_score, sub_mid_score)]))
