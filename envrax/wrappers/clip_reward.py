from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.base import EnvConfig, JaxEnv
from envrax.wrappers.base import Wrapper


class ClipReward(Wrapper):
    """
    Clip rewards to the sign of the reward: `{−1, 0, +1}`.

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
        Advance the environment by one step and clip the reward to `{−1, 0, +1}`.

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
            Observation from the inner step.
        new_state : Any
            Updated environment state.
        reward  : chex.Array
            float32 — Reward clipped to sign: `{−1.0, 0.0, +1.0}`.
        done  : chex.Array
            bool — Terminal flag from the inner step.
        info : Dict[str, Any]
            Info dict from the inner step.
        """
        obs, new_state, reward, done, info = self._env.step(rng, state, action, config)
        reward = jnp.sign(reward).astype(jnp.float32)
        return obs, new_state, reward, done, info
