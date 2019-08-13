from typing import Generic, TypeVar, Callable, Tuple, List, Dict

from distribution import *
from policy import *


_State = TypeVar('_State')
_Action = TypeVar('_Action')

Transitions = Dict[Tuple[_State, _Action], Distribution[Tuple[float, _State]]]


class System(Generic[_State, _Action]):
    def __init__(self,
                 statespace: Iterable[_State],
                 actions: Iterable[_Action],
                 dynamics: Transitions,
                 terminality: Callable[[_State], bool]=None):
        self.dynamics = dynamics
        self._terminality = terminality
        self.states = list(statespace)
        self.actions = list(actions)

    def is_terminal(self, state: _State) -> bool:
        return self._terminality(state) if self._terminality is not None else False


class MDP(Generic[_State, _Action]):
    def __init__(self,
                 system: System,
                 initial_state: _State,
                 discount: float):
        self.system = system
        self.initial_state = initial_state
        self.state = initial_state
        self.discount = discount
        self.V = None

    def reset(self, state=None):
        self.state = state if state is not None else self.initial_state

    def sample_result(self, a: _Action):
        dist = self.system.dynamics.get((self.state, a))
        if dist is None:
            raise ValueError('Illegal state/action combo')
        else:
            return dist.sample()

    def apply_policy(self, policy: Policy) -> float:
        if not self.system.is_terminal(self.state):
            action = policy(self.state).sample()
            reward, new_state = self.sample_result(self.state, action)
            self.state = new_state
            return reward
        else:
            return 0.

    def estimate_value(self, state: _State, policy: Policy, samples: int=1000, threshold: float=0.01):
        total = 0.
        for sample in range(samples):
            self.reset(state=state)
            weight = 1.
            cumulative_reward = 0.
            while (not self.system.is_terminal(self.state)) and (weight > threshold):
                cumulative_reward += weight*self.apply_policy(policy)
                weight = weight*self.discount
            total += cumulative_reward
        return total/samples

    def evaluate_policy(self,
                        policy: Policy,
                        threshold: float=0.01,
                        greedy_prob: float=0.,
                        action_stability_margin: float=0.0001,
                        round_v: Optional[int]=None) -> Dict[_State, float]:
        if self.V is None:
            self.V = {state: 0. for state in self.system.states}
        max_error = threshold*2
        iteration = 1
        policy_stable = (greedy_prob == 0.)
        while (max_error > threshold) or not policy_stable:
            stablize_policy = (max_error <= threshold)
            if stablize_policy:
                print("Stabilizing policy")
            policy_changed = False
            max_error = 0.
            for state in self.system.states:
                if not self.system.is_terminal(state):
                    action_dist = policy(state)
                    greedy = stablize_policy or (random.random() < greedy_prob)
                    if greedy:
                        best_action = None
                        best_reward = float('-inf')
                        for action in self.system.actions:
                            action_reward = expectation(self.system.dynamics.get((state, action), []),
                                                        lambda outcome: outcome[0] + self.discount*self.V[outcome[1]])
                            if action_reward > best_reward + action_stability_margin:
                                best_reward = action_reward
                                best_action = action
                        v = best_reward
                        first = next(iter(action_dist))
                        if (first.weight < 1.) or (first.value != best_action):
                            policy.update(state, best_action)
                            policy_changed = True
                    else:
                        v = 0.
                        for action in action_dist:
                            action_reward = expectation(self.system.dynamics.get((state, action.value), []),
                                                        lambda outcome: outcome[0] + self.discount*self.V[outcome[1]])
                            pi_action = action.weight
                            v += pi_action*action_reward
                else:
                    v = 0.
                error = abs(v - self.V[state])
                if error > max_error:
                    max_error = error
                self.V[state] = v if round_v is None else (round(v, round_v))
            print("After {} iterations max_error is {}".format(iteration, max_error))
            iteration += 1

            if stablize_policy and not policy_changed:
                policy_stable = True
                print("Policy is stable")
        return self.V, policy

    def optimize_policy(self,
                        initial: Policy,
                        eval_threshold: float=0.01,
                        use_value_update: bool=False,
                        greedy_prob: float=0.5,
                        action_stability_margin: float=0.0001,
                        round_v: Optional[int]=None) -> Tuple[Policy, Dict[_State,float]]:
        if use_value_update:
            V, policy = self.evaluate_policy(initial.clone(),
                                             threshold=eval_threshold,
                                             greedy_prob=greedy_prob,
                                             action_stability_margin=action_stability_margin,
                                             round_v=round_v)
        else:
            stable = False
            policy = initial.clone()
            iteration = 1
            while not stable:
                print("Commencing policy optimization iteration: {}".format(iteration))
                changed_actions = 0
                V, _ = self.evaluate_policy(policy, threshold=eval_threshold, round_v=round_v)
                stable = True
                for state in self.system.states:
                    old_action = policy(state).sample()    # sampling from deterministic policy is deterministic
                    best_val = float('-inf')
                    best_action = None
                    for action in self.system.actions:
                        action_reward = float('-inf')
                        outcomes = self.system.dynamics.get((state, action))
                        if outcomes is not None:
                            action_reward = expectation(self.system.dynamics.get((state, action)),
                                                        lambda outcome: outcome[0] + self.discount*V[outcome[1]])
                        if action_reward > best_val + action_stability_margin:
                            best_val = action_reward
                            best_action = action
                    if best_action != old_action:
                        stable = False
                        policy.update(state, best_action)
                        changed_actions += 1
                print("{} actions changed after iteration {}".format(changed_actions, iteration))
                iteration += 1

        return policy, V

    def optimize_q(self, threshold: float=0.01) -> Dict[Tuple[_State, _Action], float]:
        # start with arbitrary estimate of no value for any action
        q = {(s, a): 0. for s in self.system.states for a in self.system.actions}
        max_error = 2*threshold
        while max_error > threshold:
            max_error = 0.
            for sa in q.keys():
                if not self.system.is_terminal(sa[0]):
                    new_q = 0.
                    for outcome in self.system.dynamics.get(sa, []):
                        p = outcome.weight
                        reward = outcome.value[0]
                        next_state = outcome.value[1]
                        best_val = float('-inf')
                        for action in self.system.actions:
                            val = q.get((next_state, action), float('-inf'))
                            if val > best_val:
                                best_val = val
                        new_q += p*(reward + self.discount*best_val)
                    error = abs(q[sa] - new_q)
                    if error > max_error:
                        max_error = error
                    q[sa] = new_q
        return q

    @staticmethod
    def policy_from_q_star(q_star: Dict[Tuple[_State, _Action], float]) -> Policy:
        """Assuming a deterministic MDP return an optimal policy given optimal q*

        :param q_star: optimal q values
        :return: derived policy
        """
        best_action_vals = {}
        for (s, a), v in q_star.items():
            old = best_action_vals.get(s, (None, float('-inf')))
            if old[1] <= v:
                best_action_vals[s] = (a, v)
        action_dict = {s: Distribution.deterministic(e[0]) for s, e in best_action_vals.items()}

        return lambda s: action_dict[s]
