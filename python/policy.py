from typing import TypeVar, Generic, Callable
from abc import ABC

from distribution import *


_State = TypeVar('_State')
_Action = TypeVar('_Action')


class Policy(ABC, Generic[_State, _Action]):
    def __call__(self, state: _State) -> Distribution[_Action]:
        ...

    def update(self, state: _State, action: _Action):
        ...

    def clone(self) -> 'Policy':
        ...


class SimplePolicy(Policy[_State, _Action]):
    def __init__(self, policy_fn: Callable[[_State], Distribution[_Action]]):
        self.policy = policy_fn

    def __call__(self, state: _State):
        return self.policy(state)

    def update(self, state: _State, action: _Action):
        if not isinstance(self.policy, Policy):
            self.policy = MemoizedPolicy(self.policy)
        self.policy.update(state, action)

    def clone(self) -> 'Policy':
        return SimplePolicy(self.policy)


class MemoizedPolicy(Policy[_State, _Action]):
    def __init__(self, base_policy: Callable[[_State], Distribution[_Action]]):
        self.policy = base_policy
        self.memo = {}

    def __call__(self, state: _State):
        result = self.memo.get(state)
        if result is None:
            return self.policy(state)
        else:
            return result

    def update(self, state: _State, action: _Action):
        self.memo[state] = Distribution.deterministic(action)

    def clone(self) -> 'Policy':
        return MemoizedPolicy(self)
