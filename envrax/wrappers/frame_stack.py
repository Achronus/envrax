from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from envrax.base import EnvConfig, JaxEnv
from envrax.spaces import Box
from envrax.wrappers.base import Wrapper


@chex.dataclass
class FrameStackState:
    """
    State for `FrameStackObservation`.

    Parameters
    ----------
    env_state : Any
        Underlying environment state (may itself be a wrapped state).
    obs_stack : jax.Array
        uint8[H, W, n_stack] — Ring buffer of the last `n_stack` processed
        observations, oldest frame at channel index 0.
    """

    env_state: Any
    obs_stack: jax.Array


class FrameStackObservation(Wrapper):
    """
    Maintain a sliding window of the last `n_stack` observations.

    Expects the inner environment to produce `uint8[H, W]` observations.
    The stacked observation has shape `uint8[H, W, n_stack]`.

    Parameters
    ----------
    env : JaxEnv
        Inner environment returning 2-D observations.
    n_stack : int (optional)
        Number of frames to stack. Default is `4`.
    """

    def __init__(self, env: JaxEnv, *, n_stack: int = 4) -> None:
        super().__init__(env)
        self._n_stack = n_stack

    def reset(
        self, rng: chex.PRNGKey, config: EnvConfig
    ) -> Tuple[chex.Array, FrameStackState]:
        """
        Reset the inner environment and initialise the frame stack.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        config : EnvConfig
            Environment configuration.

        Returns
        -------
        obs  : chex.Array
            uint8[H, W, n_stack] — Initial stacked observation.
        state : FrameStackState
            Wrapper state containing the inner state and the stack.
        """
        obs, env_state = self._env.reset(rng, config)
        stack = jnp.stack([obs] * self._n_stack, axis=-1)
        wrapped = FrameStackState(env_state=env_state, obs_stack=stack)
        return stack, wrapped

    def step(
        self,
        rng: chex.PRNGKey,
        state: FrameStackState,
        action: chex.Array,
        config: EnvConfig,
    ) -> Tuple[chex.Array, FrameStackState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step the inner environment and roll the frame stack.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : FrameStackState
            Current wrapper state.
        action : chex.Array
            int32 — Action index.
        config : EnvConfig
            Environment configuration.

        Returns
        -------
        obs  : chex.Array
            uint8[H, W, n_stack] — Updated stacked observation.
        new_state : FrameStackState
            Updated wrapper state.
        reward  : chex.Array
            float32 — Reward from the inner step.
        done  : chex.Array
            bool — Terminal flag from the inner step.
        info : Dict[str, Any]
            Info dict from the inner step.
        """
        obs, env_state, reward, done, info = self._env.step(
            rng, state.env_state, action, config
        )
        new_stack = jnp.concatenate([state.obs_stack[..., 1:], obs[..., None]], axis=-1)  # type: ignore
        new_state = FrameStackState(env_state=env_state, obs_stack=new_stack)
        return new_stack, new_state, reward, done, info

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space
        h, w = inner.shape[0], inner.shape[1]
        return Box(
            low=inner.low,
            high=inner.high,
            shape=(h, w, self._n_stack),
            dtype=inner.dtype,
        )
