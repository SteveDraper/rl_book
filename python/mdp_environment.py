from typing import TypeVar, Iterable

from mdp import *
from environment import *


_State = TypeVar('_State')
_Action = TypeVar('_Action')


class MDPEnvironment(Environment[_State, _Action]):
    def __init__(self, mdp: MDP[_State, _Action]):
        self.mdp = mdp

    def reset(self, state: Optional[_State]=None):
        self.mdp.reset(state)

    def get_state(self) -> _State:
        return self.mdp.state

    def actions(self, state: Optional[_State]=None) -> Iterable[_Action]:
        return self.mdp.system.actions

    def sample(self, action: _Action) -> Tuple[float, _State, bool]:
        reward, new_state = self.mdp.sample_result(action)
        terminal = self.mdp.system.is_terminal(new_state)
        self.reset(new_state)
        return reward, new_state, terminal
