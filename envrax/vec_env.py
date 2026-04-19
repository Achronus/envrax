from pathlib import Path
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.env import EnvConfig, EnvState, JaxEnv
from envrax.spaces import Space


class VecEnv:
    """
    Wraps any `JaxEnv` to operate over a batch of environments simultaneously.

    Parameters
    ----------
    env : JaxEnv
        Single-instance environment to vectorise
    num_envs : int
        Number of parallel environments
    """

    def __init__(self, env: JaxEnv, num_envs: int) -> None:
        self.env = env
        self.num_envs = num_envs

    @property
    def config(self) -> EnvConfig:
        """Inherits configuration from the wrapped environment."""
        return self.env.config

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """
        Reset all `num_envs` environments with independent random starts.

        All returned arrays have a leading batch dimension `B = num_envs`,
        e.g. observations of shape `(B, *obs_shape)`.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs  : chex.Array
            Stacked first observations
        states : EnvState
            Batched environment states
        """
        rngs = jax.random.split(rng, self.num_envs)
        return jax.vmap(self.env.reset)(rngs)

    def step(
        self,
        state: EnvState,
        actions: chex.Array,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance all environments by one step simultaneously.

        Each environment independently auto-resets when its episode ends.
        All inputs and outputs have a leading batch dimension `B = num_envs`:

        - Observations: `(B, *obs_shape)`
        - Discrete actions: `(B,)` — one int per env
        - Continuous actions: `(B, *action_shape)` — one vector per env
        - Rewards / dones: `(B,)`

        Parameters
        ----------
        state : EnvState
            Batched environment states
        actions  : chex.Array
            One action per environment

        Returns
        -------
        obs  : chex.Array
            Observations after the step
        new_states : EnvState
            Updated batched states
        rewards  : chex.Array
            Per-environment rewards
        dones  : chex.Array
            Per-environment terminal flags
        infos : Dict[str, Any]
            Batched info dict
        """
        return jax.vmap(self._step_env)(state, actions)

    def _step_env(
        self,
        state: EnvState,
        action: chex.Array,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Single-env step that auto-resets on episode end.

        When `done` is `True`, returns the observation from a fresh reset
        instead of the terminal observation.

        Parameters
        ----------
        state : EnvState
            Current environment state
        action : chex.Array
            Action to take in the environment

        Returns
        -------
        Tuple of `(obs, new_state, reward, done, info)`, where obs/new_state
        come from reset if done is `True`.
        """
        obs, new_state, reward, done, info = self.env.step(state, action)

        reset_rng, _ = jax.random.split(new_state.rng)
        reset_obs, reset_state = self.env.reset(reset_rng)

        final_obs = jax.lax.cond(done, lambda: reset_obs, lambda: obs)
        final_state = jax.lax.cond(done, lambda: reset_state, lambda: new_state)

        return final_obs, final_state, reward, done, info

    def render(self, state: EnvState, *, index: int = 0) -> np.ndarray:
        """
        Render a single environment from the batch.

        Extracts the state at `index` from the batched state pytree and
        delegates to the inner env's `render()`.

        Parameters
        ----------
        state : EnvState
            Batched environment state
        index : int (optional)
            Which environment in the batch to render. Default is `0`.

        Returns
        -------
        frame : np.ndarray
            uint8 RGB array of shape `(H, W, 3)`
        """
        single_state = jax.tree.map(lambda x: x[index], state)
        return self.env.render(single_state)

    def compile(self, cache_dir: Path | str | None = DEFAULT_CACHE_DIR) -> None:
        """
        Trigger XLA compilation by running a dummy `reset` + `step`.

        Useful when construction and compilation should be separate phases.
        Safe to call multiple times.

        Parameters
        ----------
        cache_dir : Path | str | None (optional)
            XLA cache directory. Defaults to `~/.cache/envrax/xla_cache`.
        """
        setup_cache(cache_dir)
        _key = jax.random.key(0)
        _, _state = self.reset(_key)
        self.step(_state, jnp.zeros(self.num_envs, dtype=jnp.int32))

    @property
    def single_observation_space(self) -> Space:
        """Observation space of a single inner environment."""
        return self.env.observation_space

    @property
    def single_action_space(self) -> Space:
        """Action space of a single inner environment."""
        return self.env.action_space

    @property
    def observation_space(self) -> Space:
        """Batched observation space with a leading `num_envs` dimension."""
        return self.env.observation_space.batch(self.num_envs)

    @property
    def action_space(self) -> Space:
        """Batched action space with a leading `num_envs` dimension."""
        return self.env.action_space.batch(self.num_envs)

    def __repr__(self) -> str:
        return f"VecEnv<{self.env!r}, num_envs={self.num_envs}>"
