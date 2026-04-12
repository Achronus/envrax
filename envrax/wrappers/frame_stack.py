from typing import Any, Dict, Generic, Tuple

import chex
import jax
import jax.numpy as jnp

from envrax.env import ActSpaceT, EnvState, JaxEnv
from envrax.spaces import Box
from envrax.wrappers.base import InnerStateT, Wrapper
from envrax.wrappers.utils import require_box


@chex.dataclass
class FrameStackState(EnvState, Generic[InnerStateT]):
    """
    State for `FrameStackObservation`.

    Generic over the inner env's state type so `env_state` is precisely
    typed when the wrapper is parameterised. The base `rng`/`step`/`done`
    fields are forwarded copies from the inner state so that the framework
    (e.g. `VecEnv` auto-reset) can read them from the outer state.

    Parameters
    ----------
    env_state : InnerStateT
        Underlying environment state (may itself be a wrapped state).
    obs_stack : jax.Array
        uint8[H, W, n_stack] — Ring buffer of the last `n_stack` processed
        observations, oldest frame at channel index 0.
    """

    env_state: InnerStateT
    obs_stack: jax.Array


class FrameStackObservation(
    Wrapper[Box, ActSpaceT, FrameStackState[InnerStateT], InnerStateT]
):
    """
    Maintain a sliding window of the last `n_stack` observations.

    Expects the inner environment to produce `uint8[H, W]` observations.
    The stacked observation has shape `uint8[H, W, n_stack]`.

    Parameters
    ----------
    env : JaxEnv
        Inner environment returning 2-D `uint8` observations.
    n_stack : int (optional)
        Number of frames to stack. Default is `4`.
    """

    def __init__(
        self,
        env: JaxEnv[Box, ActSpaceT, InnerStateT],
        *,
        n_stack: int = 4,
    ) -> None:
        super().__init__(env)
        require_box(env, type(self).__name__, rank=2, dtype=jnp.uint8)
        self._n_stack = n_stack

    def reset(
        self, rng: chex.PRNGKey
    ) -> Tuple[chex.Array, FrameStackState[InnerStateT]]:
        """
        Reset the inner environment and initialise the frame stack.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs  : chex.Array
            Initial stacked observation
        state : FrameStackState
            Wrapper state containing the inner state and the stack
        """
        obs, env_state = self._env.reset(rng)
        stack = jnp.stack([obs] * self._n_stack, axis=-1)
        wrapped = FrameStackState(
            rng=env_state.rng,
            step=env_state.step,
            done=env_state.done,
            env_state=env_state,
            obs_stack=stack,
        )
        return stack, wrapped

    def step(
        self,
        state: FrameStackState[InnerStateT],
        action: chex.Array,
    ) -> Tuple[
        chex.Array,
        FrameStackState[InnerStateT],
        chex.Array,
        chex.Array,
        Dict[str, Any],
    ]:
        """
        Step the inner environment and roll the frame stack.

        Parameters
        ----------
        state : FrameStackState
            Current wrapper state
        action : chex.Array
            Action to take in the environment

        Returns
        -------
        obs  : chex.Array
            Updated stacked observation
        new_state : FrameStackState
            Updated wrapper state
        reward  : chex.Array
            Reward from the inner step
        done  : chex.Array
            Terminal flag from the inner step
        info : Dict[str, Any]
            Info dict from the inner step
        """
        obs, env_state, reward, done, info = self._env.step(state.env_state, action)
        new_stack = jnp.concatenate([state.obs_stack[..., 1:], obs[..., None]], axis=-1)
        new_state = FrameStackState(
            rng=env_state.rng,
            step=env_state.step,
            done=env_state.done,
            env_state=env_state,
            obs_stack=new_stack,
        )
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
