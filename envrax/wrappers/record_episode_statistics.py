from typing import Any, Dict, Generic, Tuple

import chex
import jax.numpy as jnp

from envrax.env import ActSpaceT, EnvState, JaxEnv, ObsSpaceT
from envrax.wrappers.base import InnerStateT, Wrapper


@chex.dataclass
class EpisodeStatisticsState(EnvState, Generic[InnerStateT]):
    """
    State for `RecordEpisodeStatistics`.

    Generic over the inner env's state type so `env_state` is precisely
    typed when the wrapper is parameterised. The base `rng`/`step`/`done`
    fields are forwarded copies from the inner state.

    Parameters
    ----------
    env_state : InnerStateT
        Inner environment state.
    episode_return : chex.Array
        Cumulative reward for the current episode. float32 scalar.
    episode_length : chex.Array
        Number of steps taken in the current episode. int32 scalar.
    """

    env_state: InnerStateT
    episode_return: chex.Array
    episode_length: chex.Array


class RecordEpisodeStatistics(
    Wrapper[ObsSpaceT, ActSpaceT, EpisodeStatisticsState[InnerStateT], InnerStateT]
):
    """
    Records episode return and length.

    Accumulates reward and step count in `EpisodeStatisticsState`.
    Episode statistics are written to `info["episode"]` on every `step()`
    call; values are non-zero only when `done=True`.

    Parameters
    ----------
    env : JaxEnv
        Environment to wrap.
    """

    def __init__(self, env: JaxEnv[ObsSpaceT, ActSpaceT, InnerStateT]) -> None:
        super().__init__(env)

    def reset(
        self, rng: chex.PRNGKey
    ) -> Tuple[chex.Array, EpisodeStatisticsState[InnerStateT]]:
        """
        Reset the environment and episode accumulators.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs  : chex.Array
            Initial observation
        state : EpisodeStatisticsState
            Initial state with zeroed accumulators
        """
        obs, env_state = self._env.reset(rng)
        state = EpisodeStatisticsState(
            rng=env_state.rng,
            step=env_state.step,
            done=env_state.done,
            env_state=env_state,
            episode_return=jnp.float32(0.0),
            episode_length=jnp.int32(0),
        )
        return obs, state

    def step(
        self,
        state: EpisodeStatisticsState[InnerStateT],
        action: chex.Array,
    ) -> Tuple[
        chex.Array,
        EpisodeStatisticsState[InnerStateT],
        chex.Array,
        chex.Array,
        Dict[str, Any],
    ]:
        """
        Step the environment and update episode accumulators.

        Parameters
        ----------
        state : EpisodeStatisticsState
            Current state
        action : chex.Array
            Action to take in the environment

        Returns
        -------
        obs  : chex.Array
            Next observation
        new_state : EpisodeStatisticsState
            Updated state
        reward  : chex.Array
            Step reward
        done  : chex.Array
            Episode terminal flag
        info : Dict[str, Any]
            Environment metadata extended with `"episode"`:
            `{"r": float32, "l": int32}` — non-zero only when `done=True`
        """
        obs, env_state, reward, done, info = self._env.step(state.env_state, action)

        episode_return = state.episode_return + reward.astype(jnp.float32)
        episode_length = state.episode_length + jnp.int32(1)

        info["episode"] = {
            "r": jnp.where(done, episode_return, jnp.float32(0.0)),
            "l": jnp.where(done, episode_length, jnp.int32(0)),
        }

        new_state = EpisodeStatisticsState(
            rng=env_state.rng,
            step=env_state.step,
            done=env_state.done,
            env_state=env_state,
            episode_return=jnp.where(done, jnp.float32(0.0), episode_return),
            episode_length=jnp.where(done, jnp.int32(0), episode_length),
        )
        return obs, new_state, reward, done, info
