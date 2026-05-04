from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.env import ActSpaceT, ConfigT, JaxEnv, ObsSpaceT, StateT
from envrax.wrappers.base import Wrapper


class ExpandDims(Wrapper[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
    """
    Add a trailing size-1 dimension to `reward` and `done`.

    Transforms scalar outputs from `step()` so that `reward` and `done`
    have shape `(..., 1)` instead of `(...)`.

    Parameters
    ----------
    env : JaxEnv
        Inner environment to wrap.
    """

    def __init__(self, env: JaxEnv[ObsSpaceT, ActSpaceT, StateT, ConfigT]) -> None:
        super().__init__(env)

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
        return self._env.reset(rng)

    def step(
        self,
        state: StateT,
        action: chex.Array,
    ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment and expand `reward` and `done`.

        Parameters
        ----------
        state : StateT
            Current environment state
        action : chex.Array
            Action to take in the environment

        Returns
        -------
        obs  : chex.Array
            Observation after the step (unchanged)
        new_state : StateT
            Updated environment state
        reward  : chex.Array
            Reward with a trailing size-1 dimension
        done  : chex.Array
            Terminal flag with a trailing size-1 dimension
        info : Dict[str, Any]
            Auxiliary info dict (unchanged)
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        return (
            obs,
            new_state,
            jnp.expand_dims(reward, -1),
            jnp.expand_dims(done, -1),
            info,
        )
