from typing import Generic, TypeVar

from environment import *
from policy import *


_State = TypeVar('_State')
_Action = TypeVar('_Action')


class MCControl(Generic[_State, _Action]):
    def __init__(self, environment: Environment[_State, _Action]):
        self.environment = environment

    def optimize_policy(self,
                        initial_policy: Policy[_State, _Action],
                        initial_epsilon: float=0.5,
                        epsilon_floor: float=0.,
                        epsilon_decay: float=1.0,
                        iterations: int=100000,
                        sample_reward_period: Optional[int]=None,
                        ucb_c: Optional[float]=None,
                        cheat: bool=False):
        epsilon = initial_epsilon
        est_policy = MemoizedPolicy(initial_policy.clone())
        q: Dict[_State, Dict[_Action, float]] = {}
        default_q_est = 0.

        def q_est(state: _State, action: _Action) -> float:
            return q.get(state, {}).get(action, default_q_est)

        def update_q_est(state: _State, action: _Action, value: float):
            state_q = q.get(state, {})
            state_q[action] = value + state_q.get(action, 0.)
            q[state] = state_q

        def greedy_action(state: _State) -> _Action:
            best = None
            best_value = float('-inf')
            for a, v in q[state].items():
                if v > best_value:
                    best_value = v
                    best = a
            return best

        count_weights: Dict[Tuple[_State, _Action], float] = {}
        counts: Dict[Tuple[_State, _Action], int] = {}

        def count(state: _State, action: _Action) -> int:
            return counts.get((state, action), 0)

        def inc_count(state: _State, action: _Action):
            counts[(state, action)] = count(state, action) + 1

        def weight(state: _State, action: _Action) -> float:
            return count_weights.get((state, action), 0.)

        def add_weight(state: _State, action: _Action, inc: float):
            count_weights[(state, action)] = weight(state, action) + inc

        total_traj_len = 0
        for i in range(iterations):
            sample_epsilon = initial_epsilon if cheat else epsilon
            if ucb_c is not None:
                ucb_policy = UCBPolicy(self.environment, count, q_est, c=ucb_c)
                exp_policy = EpsilonGreedyPolicy(self.environment, est_policy, epsilon=sample_epsilon, epsilon_policy=ucb_policy)
            else:
                exp_policy = EpsilonGreedyPolicy(self.environment, est_policy, epsilon=sample_epsilon)
            episode = self.environment.sample_episode(exp_policy)
            g = 0.
            w = 1.
            traj_len = len(episode)
            for idx in range(len(episode)):
                state, action, reward, sample_prob = episode[len(episode) - idx - 1]
                est_action = iter(est_policy(state)).__next__().value
                g += reward
                add_weight(state, action, w)
                inc_count(state, action)
                update_q_est(state, action, w*(g - q_est(state, action))/weight(state, action))
                est_policy.update(state, greedy_action(state))
                if action != est_action:
                    traj_len = idx+1
                    break
                if cheat:
                    num_actions = float(len(list(self.environment.actions(state))))
                    w/(1. - epsilon*(num_actions-1)/num_actions)
                else:
                    w = w/sample_prob
            total_traj_len += traj_len
            epsilon = max(epsilon_floor, epsilon*epsilon_decay)
            if (sample_reward_period is not None) and (i % sample_reward_period == 0):
                total = 0.
                for s_idx in range(100):
                    traj = self.environment.sample_episode(est_policy)
                    total += sum([t[2] for t in traj])
                total /= 100.
                print("After {} iterations policy value ~ {} (mean sample len {})".format(i, total, total_traj_len/sample_reward_period))
                total_traj_len = 0
        return est_policy
