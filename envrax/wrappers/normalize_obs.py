from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.env import ActSpaceT, JaxEnv, StateT
from envrax.spaces import Box
from envrax.wrappers.base import Wrapper
from envrax.wrappers.utils import require_box


class NormalizeObservation(Wrapper[Box, ActSpaceT, StateT]):
    """
    Normalises pixel observations from `uint8 [0, 255]` to `float32 [0, 1]`.

    Divides observations by 255.0 and casts to float32.

    Parameters
    ----------
    env : JaxEnv
        Environment to wrap. Must have a `Box` observation space with
        dtype `uint8`.
    """

    def __init__(self, env: JaxEnv[Box, ActSpaceT, StateT]) -> None:
        super().__init__(env)
        require_box(env, type(self).__name__, dtype=jnp.uint8)

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
        """
        Reset and return a normalised initial observation.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs  : chex.Array
            Normalised observation in `[0, 1]`
        state : StateT
            Inner environment state
        """
        obs, state = self._env.reset(rng)
        return obs.astype(jnp.float32) / jnp.float32(255.0), state

    def step(
        self,
        state: StateT,
        action: chex.Array,
    ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step and return a normalised observation.

        Parameters
        ----------
        state : StateT
            Current environment state
        action : chex.Array
            Action to take in the environment

        Returns
        -------
        obs  : chex.Array
            Normalised observation in `[0, 1]`
        new_state : StateT
            Updated environment state
        reward  : chex.Array
            Step reward
        done  : chex.Array
            Terminal flag
        info : Dict[str, Any]
            Environment metadata
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        return (
            obs.astype(jnp.float32) / jnp.float32(255.0),
            new_state,
            reward,
            done,
            info,
        )

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space
        return Box(low=0.0, high=1.0, shape=inner.shape, dtype=jnp.float32)
