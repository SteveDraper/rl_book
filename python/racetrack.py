from typing import List, Tuple, Optional, NamedTuple
from enum import Enum
from itertools import product

from toolz import curry

import matplotlib.pyplot as plt

import numpy as np

from environment import Environment
from policy import *
from mc_control import *


Point = Tuple[int, int]


def point(x: int, y: int) -> Point:
    return x, y


# stationary is a point in velocity-space
STATIONARY = point(0, 0)
# similarly for the zero iof acceleration
NO_ACC = point(0, 0)

MAX_VELOCITY = 5


EMPTY  = 0
TRACK  = 1
START  = 2
FINISH = 3


class Status(Enum):
    Ok       = 0
    CRASHED  = 1
    FINISHED = 2


TERMINAL_STATES = frozenset([Status.CRASHED, Status.FINISHED])
CRASH_EXTRA_PENALTY = 2000.
TIMEOUT_PENALTY     = 100.
DIST_REWARD_SCALE   = 2.


class HorizontalSeg(NamedTuple):
    y: int
    x_start: int
    x_end: int


class VerticalSeg(NamedTuple):
    x: int
    y_start: int
    y_end: int


class Circuit(NamedTuple):
    track: np.array
    start: HorizontalSeg
    end: VerticalSeg


class State(NamedTuple):
    position: Point
    velocity: Point
    step: int


# Actions are different accelerations which we represent as points in acceleration space
Action = Point

# valid actions only allow accelerations of -1, 0, 1
actions = [point(x, y) for x, y in product(range(-1, 2), range(-1, 2))]

MAX_STEPS = 30     # arbitrary largish number


def plot_circuit(circuit: Circuit,
                 cell_size: int = 6,
                 to_file: Optional[str]=None,
                 trajectories: List[Tuple[float, State, Action]]=None):
    max_extent = max(circuit.track.shape)
    x_bounds = (-cell_size, cell_size*(max_extent+1))
    y_bounds = (-cell_size, cell_size*(max_extent+1))

    fig=plt.figure()
    ax=fig.add_subplot(111)

    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    trajectory_colors = [
        'yellow',
        'red',
        'blue',
        'purple'
    ]

    def trajectory_color(idx: int) -> str:
        return trajectory_colors[idx % len(trajectory_colors)]

    def color(cell_type: int):
        if cell_type == TRACK:
            return 'white'
        elif cell_type == START:
            return 'brown'
        elif cell_type == FINISH:
            return 'green'
        else:
            raise ValueError("Unknown call type")

    for x in range(circuit.track.shape[0]):
        for y in range(circuit.track.shape[1]):
            if circuit.track[x, y] != EMPTY:
                polygon = plt.Rectangle((x*cell_size, y*cell_size),
                                        cell_size,
                                        cell_size,
                                        facecolor=color(circuit.track[x, y]), edgecolor='black', linewidth=1.0)
                ax.add_patch(polygon)

    if trajectories is not None:
        l = []
        for idx, trajectory in enumerate(trajectories):
            x_off = (idx%2)*(cell_size//2)
            y_off = ((idx//2)%2)*(cell_size//2)
            g = 0.
            for state, _, reward, _ in trajectory:
                x = state.position[0]
                y = state.position[1]
                polygon = plt.Rectangle((x*cell_size + x_off, y*cell_size + y_off),
                                        cell_size//2,
                                        cell_size//2,
                                        facecolor=trajectory_color(idx), edgecolor='black', linewidth=0.0)
                ax.add_patch(polygon)
                g += reward
            l.append((polygon, int(g)))
        plt.legend([e[0] for e in l], [e[1] for e in l])

    plt.box(on=None)

    if to_file:
        plt.savefig(to_file)
    else:
        plt.show()
    plt.close()


@curry
def move_status(circuit: Circuit, begin: Point, end: Point) -> Status:
    if (begin[0] < 0) or (begin[1] < 0):
        return Status.CRASHED
    elif (begin[0] >= circuit.track.shape[0]) or (begin[1] >= circuit.track.shape[1]):
        return Status.CRASHED
    elif circuit.track[begin[0], begin[1]] == EMPTY:
        return Status.CRASHED
    elif (end[0] >= 0) and\
         (end[1] >= 0) and \
         (end[0] < circuit.track.shape[0]) and \
         (end[1] < circuit.track.shape[1]) and \
         (circuit.track[end[0], end[1]] != EMPTY):
        return Status.Ok
    else:
        # interpolate back from end to start and see if the crossing point was on the finish line
        x = float(end[0])
        y = float(end[1])
        x_dist = x - begin[0]
        y_dist = y - begin[1]
        steps = max(abs(int(x_dist)), abs(int(y_dist)))
        for step in range(1, steps+1):
            x -= x_dist/steps
            y -= y_dist/steps
            x_cell = int(round(x))
            y_cell = int(round(y))
            if (x_cell >= 0) and\
                (y_cell >= 0) and\
                (x_cell < circuit.track.shape[0]) and\
                (y_cell < circuit.track.shape[1]):
                cell = circuit.track[x_cell, y_cell]
                if cell == FINISH:
                    return Status.FINISHED

        return Status.CRASHED


class RacetrackEnvironment(Environment[State, Action]):
    def __init__(self,
                 circuit: Circuit,
                 use_extra_dist_reward: bool=False,
                 no_acc_prob: float=0.):
        self.circuit = circuit
        self.state = None
        self.use_dist_reward = use_extra_dist_reward
        self.no_acc_prob = no_acc_prob
        self.reset()

    def reset(self, state: Optional[State]=None):
        if state is None:
            position = self._start_position()
            velocity = point(0, 0)
            self.state = State(position, velocity, 0)
        else:
            self.state = state

    def get_state(self) -> State:
        return self.state

    def censor_state(self, state: State) -> State:
        return State(state.position, state.velocity, 0)

    def actions(self, state: Optional[State]=None) -> Iterable[Action]:
        def legal(action: Action):
            if abs(state.velocity[0] + action[0]) > MAX_VELOCITY:
                return False
            if abs(state.velocity[1] + action[1]) > MAX_VELOCITY:
                return False
            else:
                return self._new_velocity(state.velocity, action) != STATIONARY
        return filter(legal, actions)

    def sample(self, action: Action) -> Tuple[float, State, bool]:
        if random.random() < self.no_acc_prob:
            action = point(0, 0)
        new_state = self._transition(self.state, action)
        status = move_status(self.circuit, self.state.position, new_state.position)
        if status == Status.CRASHED:
            new_state = State(self._start_position(), point(0, 0), new_state.step)
            status = Status.Ok
            reward = -CRASH_EXTRA_PENALTY
        else:
            reward = -1
        self.state = new_state
        if new_state.step > MAX_STEPS:
            status = Status.FINISHED
            reward = -TIMEOUT_PENALTY + self._dist_reward(self.state)
        return reward, new_state, status in TERMINAL_STATES

    def _dist_reward(self, state: State):
        if self.use_dist_reward:
            x_dist = float(state.position[0]) - float(self.circuit.start.x_end + self.circuit.start.x_start)/2.
            y_dist = float(self.state.position[1] - self.circuit.start.y)
            return DIST_REWARD_SCALE*math.sqrt(x_dist*x_dist + y_dist*y_dist)
        else:
            return 0.

    def _start_position(self) -> Point:
        y = self.circuit.start.y
        x = random.randint(self.circuit.start.x_start, self.circuit.start.x_end)
        return point(x, y)

    def _transition(self, state: State, action: Action):
        velocity = self._new_velocity(state.velocity, action)
        return State(point(state.position[0] + velocity[0], state.position[1] + velocity[1]),
                     velocity,
                     state.step + 1)

    def _new_velocity(self, old_velocity, action: Action):
        x_v = old_velocity[0] + action[0]
        y_v = old_velocity[1] + action[1]
        if abs(x_v) > MAX_VELOCITY:
            x_v = old_velocity[0]
        if abs(y_v) > MAX_VELOCITY:
            y_v = old_velocity[1]
        return point(x_v, y_v)


def _make_circuit(segs: List[List[Point]],
                  start: HorizontalSeg,
                  finish: VerticalSeg) -> Circuit:
    result = np.zeros((max([max([p[1] for p in l]) for l in segs]) + 1, len(segs)))
    for idx, hsegs in enumerate(segs):
        for hseg in hsegs:
            for x in range(hseg[0], hseg[1] + 1):
                result[x, idx] = TRACK

    for x in range(start.x_start, start.x_end + 1):
        result[x, start.y] = START

    for y in range(finish.y_start, finish.y_end + 1):
        result[finish.x, y] = FINISH

    return Circuit(result, start, finish)


def _straight_track(length: int, width: int) -> Circuit:
    result = np.ones((width, length))*TRACK
    result[:, 0] = START
    finish_size = min(4, result.shape[1]-1)
    result[result.shape[0]-1, result.shape[1]-finish_size:] = FINISH
    start = HorizontalSeg(0, 0, width-1)
    finish = VerticalSeg(width-1, result.shape[1]-finish_size, result.shape[1]-1)
    return Circuit(result, start, finish)


def eval_circuit(circuit: Circuit,
                 iterations: int=1000,
                 num_trajectories: int=1,
                 random_initial: bool=True,
                 use_dist_reward: bool=False,
                 cheat: bool=False,
                 perturbation: float=0.,
                 to_file: Optional[str]=None):
    run_environment = RacetrackEnvironment(circuit, use_extra_dist_reward=use_dist_reward, no_acc_prob=perturbation)

    @curry
    def random_policy_fn(environment: Environment[State, Action], state: State) -> Distribution[Action]:
        actions = list(environment.actions(state))
        return Distribution.deterministic(actions[random.randrange(len(actions))])

    @curry
    def up_and_right_slow_policy(environment: Environment[State, Action], state: State) -> Distribution[Action]:
        actions = list(environment.actions(state))
        if NO_ACC in actions:
            return Distribution.deterministic(NO_ACC)
        else:
            return Distribution.deterministic(point(1, 1))

    initial_policy = SimplePolicy((random_policy_fn if random_initial else up_and_right_slow_policy)(run_environment))
    optimizer = MCControl(run_environment)
    est_policy = optimizer.optimize_policy(initial_policy,
                                           initial_epsilon=0.5,
                                           epsilon_decay=.9995,
                                           epsilon_floor=0.075,
                                           iterations=iterations,
                                           sample_reward_period=500,
                                           ucb_c=None,
                                           cheat=cheat)
    eval_environment = RacetrackEnvironment(circuit, use_extra_dist_reward=use_dist_reward, no_acc_prob=0.)
    trajectories = [eval_environment.sample_episode(est_policy) for _ in range(num_trajectories)]
    plot_circuit(circuit, trajectories=trajectories, to_file=to_file)


if __name__ == "__main__":
    random.seed(0)
    first_circuit_segs = [
        [(3, 8)],
        [(3, 8)],
        [(3, 8)],
        [(2, 8)],
        [(2, 8)],
        [(2, 8)],
        [(2, 8)],
        [(2, 8)],
        [(2, 8)],
        [(2, 8)],
        [(1, 8)],
        [(1, 8)],
        [(1, 8)],
        [(1, 8)],
        [(1, 8)],
        [(1, 8)],
        [(1, 8)],
        [(1, 8)],
        [(0, 8)],
        [(0, 8)],
        [(0, 8)],
        [(0, 8)],
        [(0, 8)],
        [(0, 8)],
        [(0, 8)],
        [(0, 9)],
        [(0, 16)],
        [(0, 16)],
        [(1, 16)],
        [(2, 16)],
        [(2, 16)],
        [(3, 16)]
    ]
    first_circuit_start = HorizontalSeg(0, 3, 8)
    first_circuit_finish = VerticalSeg(16, len(first_circuit_segs)-6, len(first_circuit_segs)-1)

    second_circuit_segs = [
        [(0, 22)],
        [(0, 22)],
        [(0, 22)],
        [(1, 22)],
        [(2, 22)],
        [(3, 22)],
        [(4, 22)],
        [(5, 22)],
        [(6, 22)],
        [(7, 22)],
        [(8, 22)],
        [(9, 22)],
        [(10, 22)],
        [(11, 22)],
        [(12, 22)],
        [(13, 22)],
        [(13, 23)],
        [(13, 25)],
        [(13, 26)],
        [(13, 29)],
        [(12, 31)],
        [(11, 31)],
        [(10, 31)],
        [(10, 31)],
        [(10, 31)],
        [(10, 31)],
        [(11, 31)],
        [(12, 31)],
        [(15, 31)]
    ]
    second_circuit_start = HorizontalSeg(0, 0, 22)
    second_circuit_finish = VerticalSeg(31, len(second_circuit_segs)-9, len(second_circuit_segs)-1)

    circuit1 = _make_circuit(first_circuit_segs, first_circuit_start, first_circuit_finish)

    # test move status
    status = move_status(circuit1)
    assert status((0, 0), (5, 1)) == Status.CRASHED
    assert status((4, 0), (5, 1)) == Status.Ok
    assert status((4, 0), (1, len(first_circuit_segs)-1)) == Status.CRASHED
    assert status((4, 0), (4, len(first_circuit_segs))) == Status.CRASHED
    assert status((4, len(first_circuit_segs)-2), (25, len(first_circuit_segs)-2)) == Status.FINISHED

    # test_circuit = _straight_track(15, 5)
    eval_circuit(circuit1,
                 iterations=50000,
                 num_trajectories=4,
                 use_dist_reward=True,
                 perturbation=0.1,
                 to_file='circuit1.dist50k.png')

    circuit2 = _make_circuit(second_circuit_segs, second_circuit_start, second_circuit_finish)
    eval_circuit(circuit2,
                 iterations=50000,
                 num_trajectories=4,
                 use_dist_reward=True,
                 perturbation=0.1,
                 to_file='circuit2.dist50k.png')

    eval_circuit(circuit1,
                 iterations=50000,
                 num_trajectories=4,
                 use_dist_reward=False,
                 perturbation=0.1,
                 to_file='circuit1.no_dist50k.png')

    circuit2 = _make_circuit(second_circuit_segs, second_circuit_start, second_circuit_finish)
    eval_circuit(circuit2,
                 iterations=50000,
                 num_trajectories=4,
                 use_dist_reward=False,
                 perturbation=0.1,
                 to_file='circuit2.no_dist50k.png')

    eval_circuit(circuit1,
                 iterations=50000,
                 num_trajectories=4,
                 use_dist_reward=True,
                 cheat=True,
                 perturbation=0.1,
                 to_file='circuit1.dist_cheat50k.png')

    circuit2 = _make_circuit(second_circuit_segs, second_circuit_start, second_circuit_finish)
    eval_circuit(circuit2,
                 iterations=50000,
                 num_trajectories=4,
                 use_dist_reward=True,
                 cheat=True,
                 perturbation=0.1,
                 to_file='circuit2.dist_cheat50k.png')
