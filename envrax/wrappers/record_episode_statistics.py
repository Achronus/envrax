from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.base import EnvConfig, JaxEnv
from envrax.wrappers.base import Wrapper


@chex.dataclass
class EpisodeStatisticsState:
    """
    State for `RecordEpisodeStatistics`.

    Parameters
    ----------
    env_state : Any
        Inner environment state.
    episode_return : chex.Array
        Cumulative reward for the current episode. float32 scalar.
    episode_length : chex.Array
        Number of steps taken in the current episode. int32 scalar.
    """

    env_state: Any
    episode_return: chex.Array
    episode_length: chex.Array


class RecordEpisodeStatistics(Wrapper):
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

    def __init__(self, env: JaxEnv) -> None:
        super().__init__(env)

    def reset(
        self, rng: chex.PRNGKey, config: EnvConfig
    ) -> Tuple[chex.Array, EpisodeStatisticsState]:
        """
        Reset the environment and episode accumulators.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        config : EnvConfig
            Environment configuration.

        Returns
        -------
        obs  : chex.Array
            Initial observation.
        state : EpisodeStatisticsState
            Initial state with zeroed accumulators.
        """
        obs, env_state = self._env.reset(rng, config)
        state = EpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.float32(0.0),
            episode_length=jnp.int32(0),
        )
        return obs, state

    def step(
        self,
        rng: chex.PRNGKey,
        state: EpisodeStatisticsState,
        action: chex.Array,
        config: EnvConfig,
    ) -> Tuple[
        chex.Array, EpisodeStatisticsState, chex.Array, chex.Array, Dict[str, Any]
    ]:
        """
        Step the environment and update episode accumulators.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : EpisodeStatisticsState
            Current state.
        action : chex.Array
            Action to take.
        config : EnvConfig
            Environment configuration.

        Returns
        -------
        obs  : chex.Array
            Next observation.
        new_state : EpisodeStatisticsState
            Updated state.
        reward  : chex.Array
            Step reward.
        done  : chex.Array
            Episode terminal flag.
        info : Dict[str, Any]
            Environment metadata extended with `"episode"`:
            `{"r": float32, "l": int32}` — non-zero only when `done=True`.
        """
        obs, env_state, reward, done, info = self._env.step(
            rng, state.env_state, action, config
        )

        episode_return = state.episode_return + reward.astype(jnp.float32)
        episode_length = state.episode_length + jnp.int32(1)

        info["episode"] = {
            "r": jnp.where(done, episode_return, jnp.float32(0.0)),
            "l": jnp.where(done, episode_length, jnp.int32(0)),
        }

        new_state = EpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.where(done, jnp.float32(0.0), episode_return),
            episode_length=jnp.where(done, jnp.int32(0), episode_length),
        )
        return obs, new_state, reward, done, info
