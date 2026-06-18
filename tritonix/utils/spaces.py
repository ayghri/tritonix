from dataclasses import dataclass
from typing import Any
import math


@dataclass
class Choice:
    """Explicit list of allowed values."""

    options: list

    def __post_init__(self):
        self.options = list(self.options)

    def __getitem__(self, idx):
        return self.options[idx]

    def values(self) -> list[Any]:
        return list(self.options)

    def __len__(self):
        return len(self.options)


@dataclass
class Range:
    """Integers in [lo, hi] inclusive with optional step."""

    lo: int
    hi: int
    step: int = 1

    def __getitem__(self, idx):
        return self.lo + idx * self.step

    def values(self) -> list[int]:
        return list(range(self.lo, self.hi + 1, self.step))

    def __len__(self):
        return len(range(self.lo, self.hi + 1, self.step))


@dataclass
class PowerOfTwo:
    """Powers of 2 in [lo, hi] inclusive. PowerOfTwo(32, 256) -> [32, 64, 128, 256]."""

    lo: int
    hi: int

    def values(self) -> list[int]:
        vals, v = [], self.lo
        while v <= self.hi:
            vals.append(v)
            v *= 2
        return vals

    def __getitem__(self, idx):
        return self.values()[idx]

    def __len__(self):
        return len(self.values())


ConfigParam = PowerOfTwo | Range | Choice


@dataclass
class SpaceConfig:

    params: dict[str, ConfigParam]

    def __post_init__(self):
        for name, p in self.params.items():
            if len(p) == 0:
                raise ValueError(f"Param '{name}' has no values.")

    @property
    def shape(self):
        return tuple(len(p) for p in self.params.values())

    def __getitem__(self, idx_tuple: tuple[int, ...]):
        assert len(idx_tuple) == len(self.params)
        return [p[i] for p, i in zip(self.params.values(), idx_tuple)]

    def __len__(self):
        return math.prod(len(p) for p in self.params.values())
