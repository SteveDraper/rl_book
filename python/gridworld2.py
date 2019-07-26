"""
Gridworld MDPd for exercises 4.1 - 4.3
"""


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

states = [(x, y) for x, y in product(range(4), range(4))]

transitions = {}
for (x, y) in states:
    state = (x, y)
    for action in Action:
        new_state = state
        reward = -1.
        if action == Action.NORTH:
            if y > 0:
                new_state = (x, y-1)
        elif action == Action.SOUTH:
            if y < 3:
                new_state = (x, y+1)
        elif action == Action.WEST:
            if x > 0:
                new_state = (x-1, y)
        elif action == Action.EAST:
            if x < 3:
                new_state = (x+1, y)

        transitions[((x, y), action)] = Distribution.deterministic((reward, new_state))


def terminality(state: State) -> bool:
    return (state == (0, 0)) or (state == (3, 3))


def eq_prob_action():
    return [
        Weighted(action, 1./len(Action)) for action in Action
    ]


def eq_prob_random(_: State) -> Distribution[Action]:
    return Distribution(eq_prob_action)


def adjust_states42(s: Iterable[State]) -> Iterable[State]:
    return list(s) + [(1, 4)]


def adjust_transitions42(t: Transitions, include_to: bool) -> Transitions:
    """Given base transition behavior adjust to exercise (4.2) dynamics

    :param t: base transitions
    :return: adjusted transitions
    """
    new_state = (1, 4)
    new_trans = t.copy()
    if include_to:
        new_trans[((1, 3), Action.SOUTH)] = Distribution.deterministic((-1., new_state))
    new_trans[(new_state, Action.NORTH)] = Distribution.deterministic((-1., (1, 3)))
    new_trans[(new_state, Action.SOUTH)] = Distribution.deterministic((-1., (1, 4)))
    new_trans[(new_state, Action.WEST)] = Distribution.deterministic((-1., (0, 3)))
    new_trans[(new_state, Action.EAST)] = Distribution.deterministic((-1., (2, 3)))
    return new_trans


def print_v(V: Dict[State, float]):
    for s, v in V.items():
        print("\t{}: {}".format(s, round(v, 0)))


def print_q(Q: Dict[Tuple[State, Action], float]):
    for (s, a), v in Q.items():
        print("\tFrom {} {}: {}".format(s, a, round(v, 1)))


def print_policy(p: Policy):
    for s in states:
        print("\tIn {} select from {}".format(s, p(s)))


if __name__ == "__main__":
    mdp = MDP(System(states,
                     Action,
                     transitions,
                     terminality=terminality), (0, 0), 1.0)
    mdp42_1 = MDP(System(adjust_states42(states),
                         Action,
                         adjust_transitions42(transitions, include_to=False),
                         terminality=terminality), (0, 0), 1.0)
    mdp42_2 = MDP(System(adjust_states42(states),
                         Action,
                         adjust_transitions42(transitions, include_to=True),
                         terminality=terminality), (0, 0), 1.0)

    evalues, _ = mdp.evaluate_policy(eq_prob_random)
    print("Equi-probable random policy for (4.1):")
    print_v(evalues)

    evalues, _ = mdp42_1.evaluate_policy(eq_prob_random)
    print("\nEqui-probable random policy for (4.2) without trans to:")
    print_v(evalues)

    evalues, _ = mdp42_2.evaluate_policy(eq_prob_random)
    print("\nEqui-probable random policy for (4.2) with trans to:")
    print_v(evalues)

    opt_q = mdp.optimize_q()
    print("q* for (4.1):")
    print_q(opt_q)

    p_star = MDP.policy_from_q_star(opt_q)
    print("pi* for (4.1):")
    print_policy(p_star)
