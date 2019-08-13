from typing import Generic, TypeVar, Tuple, Iterable
from abc import ABC

from policy import *

_State = TypeVar('_State')
_Action = TypeVar('_Action')


class Environment(Generic[_State, _Action], ABC):
    def reset(self, state: Optional[_State]=None):
        pass

    def get_state(self) -> _State:
        pass

    def actions(self, state: Optional[_State]=None) -> Iterable[_Action]:
        pass

    def sample(self, action: _Action) -> Tuple[float, _State, bool]:
        pass

    def censor_state(self, state: _State) -> _State:
        return state

    def sample_episode(self,
                       policy: Policy,
                       start_state: Optional[_State]=None,
                       max_len: Optional[int]=None) -> List[Tuple[_State, _Action, float, float]]:
        """

        :param policy:
        :param start_state:
        :param max_len:
        :return: trajectory wherein each member is (state, action, reward, action-probability)
        """
        self.reset(start_state)
        idx = 1
        trajectory = []
        while (max_len is None) or (idx <= max_len):
            idx += 1
            state = self.get_state()
            weighted_action = policy(self.censor_state(state)).sample_weighted()
            action = weighted_action.value
            reward, new_state, terminal = self.sample(action)
            trajectory.append((self.censor_state(state), action, reward, weighted_action.weight))
            if terminal:
                break
        return trajectory


class EpsilonGreedyPolicy(Policy[_State, _Action]):
    def __init__(self,
                 environment: Environment[_State, _Action],
                 underlying: Policy[_State, _Action],
                 epsilon: float,
                 epsilon_policy: Optional[Policy]=None):
        self.environment = environment
        self.underlying = underlying
        self.epsilon = epsilon
        self.epsilon_policy = epsilon_policy or SimplePolicy(self._random_policy)

    def __call__(self, state: _State):
        dist = MixingDistribution(self.underlying(state),
                                  self.epsilon_policy(state),
                                  1. - self.epsilon,
                                  self.epsilon)
        return dist

    def _random_policy(self, state: _State) -> Distribution[_Action]:
        return Distribution.uniform(self.environment.actions(state))

    def update(self, state: _State, action: _Action):
        self.underlying.update(state, action)

    def clone(self) -> Policy[_State, _Action]:
        return EpsilonGreedyPolicy(self.environment,
                                   self.underlying,
                                   self.epsilon,
                                   self.epsilon_policy)


class UCBPolicy(Policy[_State, _Action]):
    def __init__(self,
                 environment: Environment[_State, _Action],
                 visits: Callable[[Tuple[_State, _Action]], int],
                 value_estimates: Callable[[Tuple[_State, _Action]], float],
                 c: float):
        self.environment = environment
        self.visits = visits
        self.value_estimates = value_estimates
        self.c = c

    def __call__(self, state: _State):
        actions = self.environment.actions(state)
        total_visits = 0
        action_counts = []
        for action in actions:
            count = self.visits(state, action)
            action_counts.append((action, count))
            total_visits += count
        best_action = None
        second_best_action = None
        best_raw_action = None
        best_v = float('-inf')
        second_best_v = float('-inf')
        best_raw_v = float('-inf')
        for idx, (action, count) in enumerate(action_counts):
            raw_v = self.value_estimates(state, action)
            v = self._ucb(raw_v, count, total_visits)
            if v > best_v:
                second_best_v = best_v
                second_best_action = best_action
                best_action = action
                best_v = v
            elif v > second_best_v:
                second_best_v = v
                second_best_action = action
            if v > best_raw_v:
                best_raw_v = v
                best_raw_action = action
        return Distribution.deterministic(best_action if best_action != best_raw_action else second_best_action)

    def _ucb(self, estimate: float, visits: int, parent_visits: int):
        return estimate + self.c*math.sqrt(math.log(parent_visits+1)/(visits+1))
