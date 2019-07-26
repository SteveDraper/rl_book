from typing import Tuple
from enum import Enum
from itertools import product

from distribution import *
from mdp import *


class Action(Enum):
    NORTH = 1
    SOUTH = 2
    EAST  = 3
    WEST  = 4


State = Tuple[int, int]

states = [(x, y) for x, y in product(range(5), range(5))]

transitions = {}
for (x, y) in states:
    state = (x, y)
    for action in Action:
        new_state = state
        reward = 0.
        if (x == 1) and (y == 0):
            new_state = (1, 4)
            reward = 10.
        elif (x == 3) and (y == 0):
            new_state = (3, 2)
            reward = 5.
        elif action == Action.NORTH:
            if y == 0:
                reward = -1.
            else:
                new_state = (x, y-1)
        elif action == Action.SOUTH:
            if y == 4:
                reward = -1.
            else:
                new_state = (x, y+1)
        elif action == Action.WEST:
            if x == 0:
                reward = -1.
            else:
                new_state = (x-1, y)
        elif action == Action.EAST:
            if x == 4:
                reward = -1.
            else:
                new_state = (x+1, y)

        transitions[((x, y), action)] = Distribution.deterministic((reward, new_state))


def eq_prob_action():
    return [
        Weighted(action, 1./len(Action)) for action in Action
    ]


def eq_prob_random(_: State) -> Distribution[Action]:
    return Distribution(eq_prob_action)


if __name__ == "__main__":
    mdp = MDP(System(states, Action, transitions), (0, 0), 0.9)
    # values = [[mdp.estimate_value((x, y), eq_prob_random) for x in range(5)] for y in range(5)]
    # for row in values:
    #     print(" ".join([str(v) for v in row]))
    #
    # print()

    evalues, _ = mdp.evaluate_policy(eq_prob_random)
    print("Equi-probable random policy:")
    for y in range(5):
        print(" ".join([str(evalues[(x, y)]) for x in range(5)]))

    opt_policy, _ = mdp.optimize_policy(SimplePolicy(lambda _: Distribution.deterministic(Action.NORTH)))
    opt_values, _ = mdp.evaluate_policy(opt_policy)
    print("\nOptimum policy:")
    for y in range(5):
        print(" ".join(["{}: {}".format(opt_policy((x, y)).sample(), opt_values[(x, y)]) for x in range(5)]))
