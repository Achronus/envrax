from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.env import ActSpaceT, JaxEnv, StateT
from envrax.spaces import Box
from envrax.wrappers.base import Wrapper
from envrax.wrappers.utils import require_box, resize


class ResizeObservation(Wrapper[Box, ActSpaceT, StateT]):
    """
    Resize observations to `(h, w)` using bilinear interpolation.

    Handles both:

    - **Grayscale** — `uint8[H, W]` → `uint8[h, w]`
    - **RGB** — `uint8[H, W, C]` → `uint8[h, w, C]`

    The channel dimension is preserved automatically; no pre-processing step
    is required.  For DQN-style pipelines, apply `GrayscaleObservation` first
    so the output is `uint8[h, w]` before stacking.

    Parameters
    ----------
    env : JaxEnv
        Inner environment returning `uint8[H, W]` or `uint8[H, W, C]` observations.
    h : int (optional)
        Output height in pixels. Default is `84`.
    w : int (optional)
        Output width in pixels. Default is `84`.
    """

    def __init__(
        self,
        env: JaxEnv[Box, ActSpaceT, StateT],
        *,
        h: int = 84,
        w: int = 84,
    ) -> None:
        super().__init__(env)
        require_box(env, type(self).__name__, rank=(2, 3), dtype=jnp.uint8)
        self._h = h
        self._w = w

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
        """
        Reset the inner environment and resize the observation.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs  : chex.Array
            Resized observation
        state : StateT
            Inner environment state
        """
        obs, state = self._env.reset(rng)
        return resize(obs, self._h, self._w), state

    def step(
        self,
        state: StateT,
        action: chex.Array,
    ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step the inner environment and resize the observation.

        Parameters
        ----------
        state : StateT
            Current environment state
        action : chex.Array
            Action to take in the environment

        Returns
        -------
        obs  : chex.Array
            Resized observation
        new_state : StateT
            Updated environment state
        reward  : chex.Array
            Reward from the inner step
        done  : chex.Array
            Terminal flag from the inner step
        info : Dict[str, Any]
            Info dict from the inner step
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        return resize(obs, self._h, self._w), new_state, reward, done, info

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space

        if len(inner.shape) == 3:
            shape = (self._h, self._w, inner.shape[-1])
        else:
            shape = (self._h, self._w)

        return Box(low=inner.low, high=inner.high, shape=shape, dtype=inner.dtype)
