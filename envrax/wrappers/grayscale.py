from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.base import ActSpaceT, JaxEnv, StateT
from envrax.spaces import Box
from envrax.wrappers.base import Wrapper
from envrax.wrappers.utils import require_box, to_gray


class GrayscaleObservation(Wrapper[Box, ActSpaceT, StateT]):
    """
    Convert RGB observations to grayscale using the NTSC luminance formula.

    Wraps any environment whose `reset` / `step` return `uint8[H, W, 3]`
    observations and converts them to `uint8[H, W]`.

    Parameters
    ----------
    env : JaxEnv
        Inner environment to wrap. Must have a `Box` observation space
        of shape `(H, W, 3)` and dtype `uint8`.
    """

    def __init__(self, env: JaxEnv[Box, ActSpaceT, StateT]) -> None:
        super().__init__(env)
        require_box(
            env,
            type(self).__name__,
            rank=3,
            last_dim=3,
            dtype=jnp.uint8,
        )

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
        """
        Reset the inner environment and convert the observation to grayscale.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs  : chex.Array
            Grayscale observation
        state : StateT
            Inner environment state
        """
        obs, state = self._env.reset(rng)
        return to_gray(obs), state

    def step(
        self,
        state: StateT,
        action: chex.Array,
    ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step the inner environment and convert the observation to grayscale.

        Parameters
        ----------
        state : StateT
            Current environment state
        action : chex.Array
            Action to take in the environment

        Returns
        -------
        obs  : chex.Array
            Grayscale observation
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
        return to_gray(obs), new_state, reward, done, info

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space
        h, w = inner.shape[0], inner.shape[1]
        return Box(low=inner.low, high=inner.high, shape=(h, w), dtype=inner.dtype)
