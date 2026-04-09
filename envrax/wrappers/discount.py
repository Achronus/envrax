from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.base import EnvParams, JaxEnv
from envrax.wrappers.base import Wrapper


class EpisodeDiscount(Wrapper):
    """
    Convert the boolean `done` signal to a float32 episode discount.

    The 4th return value of `step()` changes from `bool` to `float32`:
    `1.0` while the episode is running, `0.0` on termination.

    Parameters
    ----------
    env : JaxEnv
        Inner environment to wrap.
    """

    def __init__(self, env: JaxEnv) -> None:
        super().__init__(env)

    def reset(self, rng: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, Any]:
        return self._env.reset(rng, params)

    def step(
        self,
        rng: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment and return a float32 discount instead of done.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : Any
            Current environment state.
        action : chex.Array
            int32 — Action index.
        params : EnvParams
            Environment parameters.

        Returns
        -------
        obs  : chex.Array
            Observation from the inner step.
        new_state : Any
            Updated environment state.
        reward  : chex.Array
            float32 — Reward from the inner step (unchanged).
        discount  : chex.Array
            float32 — `1.0` if the episode continues, `0.0` if it terminated.
        info : Dict[str, Any]
            Info dict from the inner step.
        """
        obs, new_state, reward, done, info = self._env.step(rng, state, action, params)
        discount = jnp.where(done, jnp.float32(0.0), jnp.float32(1.0))
        return obs, new_state, reward, discount, info
