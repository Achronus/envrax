from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from envrax.env import ActSpaceT, ConfigT, JaxEnv, ObsSpaceT, StateT
from envrax.wrappers.base import Wrapper


class ClipReward(Wrapper[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
    """
    Clip rewards to the sign of the reward: `{−1, 0, +1}`.

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
        Advance the environment by one step and clip the reward to `{−1, 0, +1}`.

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
            Reward clipped to sign: `{−1.0, 0.0, +1.0}`
        done  : jax.Array
            Terminal flag from the inner step
        info : Dict[str, Any]
            Info dict from the inner step
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        reward = jnp.sign(reward).astype(jnp.float32)
        return obs, new_state, reward, done, info
