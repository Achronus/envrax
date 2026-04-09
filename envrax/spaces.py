from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import chex
import jax
import jax.numpy as jnp


class Space(ABC):
    """Abstract base class for action and observation spaces."""

    @abstractmethod
    def sample(self, rng: chex.Array) -> chex.Array:
        """Sample a random element from this space."""
        ...

    @abstractmethod
    def contains(self, x: chex.Array) -> bool:
        """Return True if x is a valid element of this space."""
        ...


@dataclass(frozen=True)
class Discrete(Space):
    """
    Discrete action space — `n` equally-likely integer actions.

    Parameters
    ----------
    n : int
        Number of discrete actions.
    """

    n: int

    def sample(self, rng: chex.Array) -> chex.Array:
        """
        Sample a random action uniformly from `[0, n)`.

        Parameters
        ----------
        rng : chex.Array
            JAX PRNG key.

        Returns
        -------
        action : chex.Array
            int32 — Sampled action index.
        """
        return jax.random.randint(
            rng, shape=(), minval=0, maxval=self.n, dtype=jnp.int32
        )

    def contains(self, x: chex.Array) -> bool:
        return bool((x >= 0) & (x < self.n))


@dataclass(frozen=True)
class Box(Space):
    """
    Continuous box observation space with scalar bounds.

    Parameters
    ----------
    low : float
        Lower bound (inclusive) applied to all elements.
    high : float
        Upper bound (inclusive) applied to all elements.
    shape : Tuple[int, ...]
        Shape of a single observation.
    dtype : type
        Element dtype. Defaults to `jnp.float32`.
    """

    low: float
    high: float
    shape: Tuple[int, ...]
    dtype: type = jnp.float32

    def sample(self, rng: chex.Array) -> chex.Array:
        """
        Sample a random observation within `[low, high]`.

        Parameters
        ----------
        rng : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            `dtype[*shape]` — Sampled observation array.
        """
        if jnp.issubdtype(self.dtype, jnp.integer):
            return jax.random.randint(
                rng,
                shape=self.shape,
                minval=int(self.low),
                maxval=int(self.high) + 1,
                dtype=self.dtype,
            )
        return jax.random.uniform(
            rng, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: chex.Array) -> bool:
        return bool(
            (x.shape == self.shape) & jnp.all(x >= self.low) & jnp.all(x <= self.high)
        )
