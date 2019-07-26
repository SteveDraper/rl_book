from typing import Generic, TypeVar, Iterable, Tuple, Callable, List, Optional

from itertools import product

import numpy as np
import random
import math


_A = TypeVar('_A')
_B = TypeVar('_B')


class Weighted(Generic[_A]):
    def __init__(self, a: _A, weight: float):
        self.value = a
        self.weight = weight

    def map_weight(self, f: Callable[[float], float]) -> 'Weighted[_A]':
        return Weighted(self.value, f(self.weight))


class Distribution(Generic[_A]):
    def __init__(self, probs: Callable[[], Iterable[Weighted[_A]]], normalize=True):
        self.prob_fn = probs
        if normalize:
            self.total = 0.
            for e in probs():
                self.total += e.weight
        else:
            self.total = None

    def __iter__(self):
        if self.total is not None:
            for e in self.prob_fn():
                yield e.map_weight(lambda p: p/self.total)
        else:
            yield from self.prob_fn()

    def __str__(self):
        return str({e.value: e.weight for e in self.prob_fn()})

    def sample(self) -> _A:
        p = random.random()
        for e in self:
            if e.weight > p:
                return e.value
            else:
                p -= e.weight

        assert False

    @classmethod
    def deterministic(cls, value: _A) -> 'Distribution[_A]':
        return cls(lambda: [Weighted(value, 1.)])


class StrictDistribution(Distribution[_A]):
    def __init__(self,
                 probs: Callable[[], Iterable[Weighted[_A]]],
                 normalize=True,
                 trim_threshold: Optional[float]=None):
        strict = list(probs())
        values = [e.value for e in strict]
        probs = np.array([e.weight for e in strict])
        if trim_threshold is not None:
            perm = np.argsort(-probs)
            total = 0.
            self.values = []
            top_probs = []
            for idx in range(len(probs)):
                top_probs.append(probs[perm[idx]])
                self.values.append(values[perm[idx]])
                total += probs[perm[idx]]
                if total >= trim_threshold:
                    break
            self.probs = (np.array(top_probs)/total).reshape(len(self.values), 1)
        else:
            self.probs = probs.reshape(len(values), 1)
            self.values = values

        super(StrictDistribution, self).__init__(self._outcomes, normalize=normalize)

    def _outcomes(self):
        for idx, v in enumerate(self.values):
            yield Weighted(v, float(self.probs[idx]))


class _StrictProduct(Distribution[Tuple[_A, _B]]):
    def __init__(self,
                 d1: StrictDistribution[_A],
                 d2: StrictDistribution[_B]):
        self.d1 = d1
        self.d2 = d2
        self.probs = np.matmul(d1.probs, np.transpose(d2.probs))
        self.total = None
        super(_StrictProduct, self).__init__(self._outcomes)

    def _outcomes(self):
        for idx, p in np.ndenumerate(self.probs):
            yield Weighted((self.d1.values[idx[0]], self.d2.values[idx[1]]), p)


class JointDistribution(Distribution[Tuple[_A, _B]]):
    def __init__(self,
                 d1: Distribution[_A],
                 d2: Distribution[_B]):
        if isinstance(d1, StrictDistribution) and isinstance(d2, StrictDistribution):
            self.prod = _StrictProduct(d1, d2)
            self.prob_fn = lambda: self.prod
        else:
            self.prob_fn = self._naive_outcomes(d1, d2)
        self.total = None

    @staticmethod
    def _naive_outcomes():
        for (outcome1, outcome2) in product(d1, d2):
            yield Weighted((outcome1.value, outcome2.value), outcome1.weight*outcome2.weight)


def dist_product(d1: Distribution[_A],
                 d2: Distribution[_B]) -> Distribution[Tuple[_A, _B]]:
    return JointDistribution(d1, d2)


def Poisson(lmda: float, max_val: int=20, discard_threshold=0.001) -> Distribution[int]:
    def generator():
        total = 0.
        n_bang = 1.
        p = 1.
        for n in range(max_val):
            p = math.exp(-lmda)*math.pow(lmda, n)/n_bang
            if (p < discard_threshold) and (n > lmda):
                break
            n_bang *= (n+1)
            total += p
            yield Weighted(n, p)

        assert p <= 1.

        # normalize by considering the rest to just be the max value + 1
        yield Weighted(max_val, 1. - total)

    return StrictDistribution(generator, normalize=False)


