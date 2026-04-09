from typing import Any, Dict, Tuple

import chex

from envrax.base import EnvConfig, JaxEnv
from envrax.spaces import Box
from envrax.wrappers.base import Wrapper
from envrax.wrappers.utils import to_gray


class GrayscaleObservation(Wrapper):
    """
    Convert RGB observations to grayscale using the NTSC luminance formula.

    Wraps any environment whose `reset` / `step` return `uint8[H, W, 3]`
    observations and converts them to `uint8[H, W]`.

    Parameters
    ----------
    env : JaxEnv
        Inner environment to wrap.
    """

    def __init__(self, env: JaxEnv) -> None:
        super().__init__(env)

    def reset(self, rng: chex.PRNGKey, config: EnvConfig) -> Tuple[chex.Array, Any]:
        """
        Reset the inner environment and convert the observation to grayscale.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        config : EnvConfig
            Environment configuration.

        Returns
        -------
        obs  : chex.Array
            uint8[H, W] — Grayscale observation.
        state : Any
            Inner environment state.
        """
        obs, state = self._env.reset(rng, config)
        return to_gray(obs), state

    def step(
        self,
        rng: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        config: EnvConfig,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step the inner environment and convert the observation to grayscale.

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
            uint8[H, W] — Grayscale observation.
        new_state : Any
            Updated environment state.
        reward  : chex.Array
            float32 — Reward from the inner step.
        done  : chex.Array
            bool — Terminal flag from the inner step.
        info : dict
            Info dict from the inner step.
        """
        obs, new_state, reward, done, info = self._env.step(rng, state, action, config)
        return to_gray(obs), new_state, reward, done, info

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space
        h, w = inner.shape[0], inner.shape[1]
        return Box(low=inner.low, high=inner.high, shape=(h, w), dtype=inner.dtype)
