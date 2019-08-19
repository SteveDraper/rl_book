"""
Windy gridworld MDPd for exercise 6.9
"""


from typing import Tuple, List
from enum import Enum
from itertools import product

from toolz import curry

from distribution import *
from mdp import *
from mdp_environment import MDPEnvironment
from sarsa import *


Action = Tuple[int, int]

actions = [
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1)
]

additional_king_actions = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1)
]

additional_stationary_actions = [
    (0, 0)
]

State = Tuple[int, int]

MAX_X = 9
MAX_Y = 6

states = [(x, y) for x, y in product(range(MAX_X+1), range(MAX_Y+1))]
winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


@curry
def clamp(min: int, max: int, value: int) -> int:
    if value < min:
        return min
    elif value > max:
        return max
    else:
        return value

clamp_x = clamp(0, MAX_X)
clamp_y = clamp(0, MAX_Y)


def transitions(stochastic_wind: bool, kings: bool, stationary: bool) -> Dict[State, Distribution[Action]]:
    t = {}
    a = actions
    if kings:
        a += additional_king_actions
    if stationary:
        a += additional_stationary_actions

    for (x, y) in states:
        state = (x, y)
        for action in a:
            new_state = (state[0] + action[0], state[1] + action[1])
            reward = -1.

            if stochastic_wind:
                y_possibilities = [clamp_y(new_state[1]-winds[state[0]]+s) for s in range(-1,2)]
                possibilities = [(reward, (clamp_x(new_state[0]), y)) for y in y_possibilities]
                t[((x, y), action)] = Distribution.uniform(possibilities)
            else:
                new_state = (clamp_x(new_state[0]), clamp_y(new_state[1]-winds[state[0]]))
                t[((x, y), action)] = Distribution.deterministic((reward, new_state))
    return t


def initial_q(state: State, _: Action) -> float:
    return 0. if state == (7, 3) else -5.


def terminality(state: State) -> bool:
    return state == (7, 3)


def print_trajectory(traj: List[Tuple[State, Action]], name: str):
    print("Trajectory for " + name)
    for t in traj:
        print("\t{} select {}".format(t[0], t[1]))


def solve(stochastic_wind: bool,
          kings: bool,
          stationary: bool,
          name: str,
          num_trajectories: int=1):
    mdp = MDP(System(states,
                     actions,
                     transitions(stochastic_wind, kings, stationary),
                     terminality=terminality), (0, 3), 1.0)
    env = MDPEnvironment(mdp)
    policy = sarsa(env,
                   default_q_est=initial_q,
                   initial_epsilon=0.5,
                   epsilon_decay=0.995,
                   iterations=3000)
    for idx in range(1, num_trajectories+1):
        traj = env.sample_episode(policy)
        print_trajectory([(t[0], t[1]) for t in traj], name="{} ({})".format(name, idx))


if __name__ == "__main__":
    solve(False, False, False, "base")
    solve(False, True, False, "kings moves")
    solve(False, True, True, "kings moves + stationary")
    solve(True, True, False, "kings moves + stochastic wind", num_trajectories=5)

