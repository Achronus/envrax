from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Type

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

    @abstractmethod
    def batch(self, n: int) -> "Space":
        """
        Return a batched version of this space with a leading dimension `n`.

        Parameters
        ----------
        n : int
            Batch size.

        Returns
        -------
        batched : Space
            Space with a leading `n` dimension.
        """
        ...


@dataclass(frozen=True)
class Discrete(Space):
    """
    Discrete action space — `n` equally-likely integer actions.

    Parameters
    ----------
    n : int
        Number of discrete actions.
    dtype : Type
        Element dtype. Defaults to `jnp.int32`.
    """

    n: int
    dtype: Type = jnp.int32

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
            rng,
            shape=(),
            minval=0,
            maxval=self.n,
            dtype=self.dtype,
        )

    def contains(self, x: chex.Array) -> bool:
        """
        Return True if `x` is a valid action index.

        Parameters
        ----------
        x : chex.Array
            Action to validate. Expected to be an integer scalar.

        Returns
        -------
        valid : bool
            True if `x` lies in `[0, n)`, False otherwise.
        """
        return bool((x >= 0) & (x < self.n))

    def batch(self, n: int) -> "MultiDiscrete":
        """
        Batch `n` copies into a `MultiDiscrete` with identical sub-spaces.

        Parameters
        ----------
        n : int
            Batch size.

        Returns
        -------
        batched : MultiDiscrete
            `MultiDiscrete(nvec=(self.n,) * n, dtype=self.dtype)`.
        """
        return MultiDiscrete(nvec=(self.n,) * n, dtype=self.dtype)


@dataclass(frozen=True)
class Box(Space):
    """
    Continuous box observation space with scalar bounds.

    Parameters
    ----------
    low : float | int
        Lower bound (inclusive) applied to all elements.
    high : float | int
        Upper bound (inclusive) applied to all elements.
    shape : Tuple[int, ...]
        Shape of a single observation.
    dtype : Type
        Element dtype. Defaults to `jnp.float32`.
    """

    low: float | int
    high: float | int
    shape: Tuple[int, ...]
    dtype: Type = jnp.float32

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
            rng,
            shape=self.shape,
            minval=self.low,
            maxval=self.high,
        ).astype(self.dtype)

    def contains(self, x: chex.Array) -> bool:
        """
        Return True if `x` is a valid observation within the space.

        Parameters
        ----------
        x : chex.Array
            Observation to validate. Expected to match `self.shape`.

        Returns
        -------
        valid : bool
            True if `x.shape == self.shape` and every element lies in `[low, high]`.
        """
        return bool(
            (x.shape == self.shape) & jnp.all(x >= self.low) & jnp.all(x <= self.high)
        )

    def batch(self, n: int) -> "Box":
        """
        Prepend a leading `n` dimension to the shape.

        Parameters
        ----------
        n : int
            Batch size.

        Returns
        -------
        batched : Box
            `Box` with shape `(n, *self.shape)` and unchanged bounds/dtype.
        """
        return Box(
            low=self.low,
            high=self.high,
            shape=(n, *self.shape),
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class MultiDiscrete(Space):
    """
    A vector of independent discrete actions,
    each with its own number of options.

    Parameters
    ----------
    nvec : Tuple[int, ...]
        Number of actions for each discrete sub-space.
    dtype : Type
        Element dtype. Defaults to `jnp.int32`.
    """

    nvec: Tuple[int, ...]
    dtype: Type = jnp.int32

    def sample(self, rng: chex.Array) -> chex.Array:
        """
        Sample one action per sub-space.

        Parameters
        ----------
        rng : chex.Array
            JAX PRNG key.

        Returns
        -------
        actions : chex.Array
            `int32[len(nvec)]` — One sampled action per sub-space.
        """
        nvec_arr = jnp.array(self.nvec, dtype=self.dtype)
        return jax.random.randint(
            rng,
            shape=(len(self.nvec),),
            minval=0,
            maxval=nvec_arr,
            dtype=self.dtype,
        )

    def contains(self, x: chex.Array) -> bool:
        """
        Return True if `x` is a valid multi-discrete action vector.

        Parameters
        ----------
        x : chex.Array
            Action vector to validate. Expected to have shape `(len(nvec),)`.

        Returns
        -------
        valid : bool
            True if `x` has shape `(len(nvec),)` and each `x[i]` is in `[0, nvec[i])`.
        """
        if x.shape != (len(self.nvec),):
            return False

        nvec_arr = jnp.array(self.nvec, dtype=self.dtype)
        return bool(jnp.all(x >= 0) & jnp.all(x < nvec_arr))

    def batch(self, n: int) -> "MultiDiscrete":
        """
        Repeat `nvec` `n` times to form a wider `MultiDiscrete`.

        Parameters
        ----------
        n : int
            Batch size.

        Returns
        -------
        batched : MultiDiscrete
            `MultiDiscrete(nvec=self.nvec * n, dtype=self.dtype)`.
        """
        return MultiDiscrete(
            nvec=self.nvec * n,
            dtype=self.dtype,
        )
