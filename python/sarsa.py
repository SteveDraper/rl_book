from typing import Generic, TypeVar, Tuple, Callable, Dict, Optional

from toolz import compose

from environment import *
from policy import *


_State = TypeVar('_State')
_Action = TypeVar('_Action')


def sarsa(env: Environment[_State, _Action],
          iterations: int=5000,
          alpha: float=0.1,
          discount: float=1.,
          default_q_est: Optional[Callable[[Tuple[_State, _Action]], float]]=None,
          report_period: int=100,
          initial_epsilon: float=0.1,
          epsilon_decay: float=1.) -> Policy[_State, _Action]:
    q: Dict[_State, Dict[_Action, float]] = {}

    def q_est(state: _State, action: _Action) -> float:
        state_q = q.get(state, {})
        value = state_q.get(action)
        if value is None:
            value = default_q_est(state, action) if default_q_est is not None else 0.
        return value

    def update_q_est(state: _State, action: _Action, value: float):
        state_q = q.get(state, {})
        state_q[action] = value + q_est(state, action)
        q[state] = state_q

    def greedy_action(state: _State) -> _Action:
        best = None
        best_value = float('-inf')
        actions = env.actions(state)
        for a in actions:
            v = q_est(state, a)
            if v > best_value:
                best_value = v
                best = a
        return best

    greedy_policy = SimplePolicy(compose(Distribution.deterministic, greedy_action))

    epsilon = initial_epsilon
    total_len = 0
    for iteration in range(iterations):
        exp_policy = EpsilonGreedyPolicy(env, greedy_policy, epsilon)
        episode_len = 0
        env.reset()
        state = env.get_state()
        action = exp_policy(state).sample()
        while True:
            reward, new_state, terminal = env.sample(action)
            episode_len += 1
            next_action = exp_policy(new_state).sample()
            delta_q = alpha*(reward + discount*q_est(new_state, next_action) - q_est(state, action))
            update_q_est(state, action, delta_q)
            state = new_state
            action = next_action
            if terminal:
                total_len += episode_len
                break
        if iteration % report_period == report_period-1:
            print("Mean episode length after {} iterations: {}".format(iteration+1,
                                                                       float(total_len)/float(report_period)))
            total_len = 0
        epsilon *= epsilon_decay

    return greedy_policy
