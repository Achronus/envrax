from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from envrax.env import ActSpaceT, ConfigT, JaxEnv, ObsSpaceT, StateT
from envrax.wrappers.base import Wrapper


class EpisodeDiscount(Wrapper[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
    """
    Convert the boolean `done` signal to a float32 episode discount.

    The 4th return value of `step()` changes from `bool` to `float32`:
    `1.0` while the episode is running, `0.0` on termination.

    Parameters
    ----------
    env : JaxEnv
        Inner environment to wrap.
    """

    def __init__(self, env: JaxEnv[ObsSpaceT, ActSpaceT, StateT, ConfigT]) -> None:
        super().__init__(env)

    def reset(self, rng: chex.PRNGKey) -> Tuple[jax.Array, StateT]:
        return self._env.reset(rng)

    def step(
        self,
        state: StateT,
        action: jax.Array,
    ) -> Tuple[jax.Array, StateT, jax.Array, jax.Array, Dict[str, Any]]:
        """
        Advance the environment and return a float32 discount instead of done.

        Parameters
        ----------
        state : StateT
            Current environment state
        action : jax.Array
            Action to take in the environment

        Returns
        -------
        obs  : jax.Array
            Observation from the inner step
        new_state : StateT
            Updated environment state
        reward  : jax.Array
            Reward from the inner step (unchanged)
        discount  : jax.Array
            `1.0` if the episode continues, `0.0` if it terminated
        info : Dict[str, Any]
            Info dict from the inner step
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        discount = jnp.where(done, jnp.float32(0.0), jnp.float32(1.0))
        return obs, new_state, reward, discount, info
