"""
Jack's car rental MDPd for exercises 4.7
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
Action = int    # strictly from range(-5,6)

actions = list(range(-5, 6))

# State is (cars at first, cars at second)
State = Tuple[int, int]

MAX_CARS_AT_LOC = 20

states = [(x, y) for x, y in product(range(MAX_CARS_AT_LOC+1), range(MAX_CARS_AT_LOC+1))]
clip = clamp(0, MAX_CARS_AT_LOC)


# Pre-calculate the Poisson joints we'll need
returns_dists = [[None]*(MAX_CARS_AT_LOC+1) for _ in range(MAX_CARS_AT_LOC+1)]
for (max1, max2) in product(range(MAX_CARS_AT_LOC+1), range(MAX_CARS_AT_LOC+1)):
    returns_dists[max1][max2] = dist_product(Poisson(3, max_val=max1), Poisson(2, max_val=max2))

request_dists = [[None]*(MAX_CARS_AT_LOC+1) for _ in range(MAX_CARS_AT_LOC+1)]
for (max1, max2) in product(range(MAX_CARS_AT_LOC+1), range(MAX_CARS_AT_LOC+1)):
    request_dists[max1][max2] = dist_product(Poisson(3, max_val=max1), Poisson(4, max_val=max2))


class OutcomeDistribution(Distribution[Tuple[float, State]]):
    def __init__(self, state: State, action: Action, modified_dynamics: bool=False):
        (count1, count2) = state

        # available to rent is end of day + impact of moved vehicles
        self.available_loc1 = clip(count1 - action)
        self.available_loc2 = clip(count2 + action)
        self.requests_dist = request_dists[self.available_loc1][self.available_loc2]
        self.total = None
        self.action = action
        self.prob_fn = lambda: self._outcomes(modified_dynamics)

    def _outcomes(self, modified_dynamics: bool):
        for requests in self.requests_dist:
            (requests_loc1, requests_loc2) = requests.value
            max_returns_loc1 = MAX_CARS_AT_LOC - self.available_loc1 + requests_loc1
            max_returns_loc2 = MAX_CARS_AT_LOC - self.available_loc2 + requests_loc2

            # calc reward and new state
            rented_loc1 = min(self.available_loc1, requests_loc1)
            rented_loc2 = min(self.available_loc2, requests_loc2)
            rental_reward = 10.*(rented_loc1 + rented_loc2)

            for returns in returns_dists[max_returns_loc1][max_returns_loc2]:
                (returns_loc1, returns_loc2) = returns.value
                p = returns.weight*requests.weight

                end_count1 = clip(self.available_loc1 - rented_loc1 + returns_loc1)
                end_count2 = clip(self.available_loc2 - rented_loc2 + returns_loc2)

                if modified_dynamics:
                    action_cost = 2.*-self.action if self.action <= 0 else 2.*(self.action-1)
                    parking_cost = 0. if (end_count2 < 10) and (end_count1 < 10) else 4.
                    reward = rental_reward - action_cost + parking_cost
                else:
                    reward = rental_reward - 2.*abs(self.action)

                yield Weighted((reward, (end_count1, end_count2)), p)


def transitions(modified_dynamics: bool=False):
    result = {}
    for s in states:
        for action in actions:
            # trim out illegal actions
            if (action <= s[0]) and (-action <= s[1]):
                result[(s, action)] = StrictDistribution(lambda: OutcomeDistribution(s,
                                                                                     action,
                                                                                     modified_dynamics=modified_dynamics),
                                                                                     trim_threshold=0.999)
                # print("State {}, action {} has {} outcomes".format(s, action, len(result[(s, action)].probs)))
        print("Calculated transitions for {}".format(s))
    return result


def print_v(V: Dict[State, float]):
    for s, v in V.items():
        print("\t{}: {}".format(s, round(v, 0)))


def print_q(Q: Dict[Tuple[State, Action], float]):
    for (s, a), v in Q.items():
        print("\tFrom {} {}: {}".format(s, a, round(v, 1)))


def print_policy(p: Policy):
    for s in states:
        print("\tIn {} select from {}".format(s, p(s)))


def plot_policy(policy: Policy, title: str, name: str):
    x = np.arange(0, 21)
    y = np.arange(0, 21)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for idx1 in range(X.shape[0]):
        for idx2 in range(X.shape[1]):
            z = policy((Y[idx1, idx2], X[idx1, idx2])).sample()
            Z[idx1, idx2] = z
    fig, ax = plt.subplots()
    plt.xticks(x)
    plt.yticks(y)
    CS = ax.contour(X, Y, Z, levels=actions)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title(title)
    plt.savefig(name + "_contour.png")
    plt.close()

    fig, ax = plt.subplots()
    plt.xticks(x)
    plt.yticks(y)
    ax.matshow(Z, vmin=-5, vmax=5, origin='lower')
    ax.set_title(title)
    plt.savefig(name + "_heatmap.png")
    plt.close()


if __name__ == "__main__":
    mdp = MDP(System(states,
                     actions,
                     transitions()), (0, 0), 0.9)

    policy = MemoizedPolicy(lambda _: Distribution.deterministic(0))
    p_star, v = mdp.optimize_policy(policy,
                                    eval_threshold=0.25,
                                    use_value_update=True,
                                    greedy_prob=0.25)
    print("pi* for (4.7) base dynamics:")
    print_policy(p_star)
    print_v(v)
    plot_policy(p_star, "Base dynamics", "base_dynamics_vi")

    policy = MemoizedPolicy(lambda _: Distribution.deterministic(0))
    mdp = MDP(System(states,
                     actions,
                     transitions(modified_dynamics=True)), (0, 0), 0.9)
    p_star, v = mdp.optimize_policy(policy,
                                    eval_threshold=0.25,
                                    use_value_update=True,
                                    greedy_prob=0.25)
    print("pi* for (4.7) modified dynamics:")
    print_policy(p_star)
    plot_policy(p_star, "Modified dynamics", "modified_dynamics_vi")
