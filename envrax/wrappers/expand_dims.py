from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.base import EnvConfig, JaxEnv
from envrax.wrappers.base import Wrapper


class ExpandDims(Wrapper):
    """
    Add a trailing size-1 dimension to `reward` and `done`.

    Transforms scalar outputs from `step()` so that `reward` and `done`
    have shape `(..., 1)` instead of `(...)`.

    Parameters
    ----------
    env : JaxEnv
        Inner environment to wrap.
    """

    def __init__(self, env: JaxEnv) -> None:
        super().__init__(env)

    def reset(self, rng: chex.PRNGKey, config: EnvConfig) -> Tuple[chex.Array, Any]:
        return self._env.reset(rng, config)

    def step(
        self,
        rng: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        config: EnvConfig,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment and expand `reward` and `done`.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : Any
            Current environment state.
        action : chex.Array
            int32 — Action index.
        config : EnvConfig
            Environment configuration.

        Returns
        -------
        obs  : chex.Array
            Observation after the step (unchanged).
        new_state : Any
            Updated environment state.
        reward  : chex.Array
            float32[..., 1] — Reward with a trailing size-1 dimension.
        done  : chex.Array
            bool[..., 1] — Terminal flag with a trailing size-1 dimension.
        info : Dict[str, Any]
            Auxiliary info dict (unchanged).
        """
        obs, new_state, reward, done, info = self._env.step(rng, state, action, config)
        return (
            obs,
            new_state,
            jnp.expand_dims(reward, -1),
            jnp.expand_dims(done, -1),
            info,
        )
