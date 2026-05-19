from pathlib import Path
from typing import Any, Dict, Generic, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.batched_env import BatchedEnv
from envrax.env import ActSpaceT, ConfigT, JaxEnv, ObsSpaceT, StateT
from envrax.spaces import Space


class VecEnv(BatchedEnv, Generic[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
    """
    Wraps any `JaxEnv` to operate over a batch of environments simultaneously.

    Canonical `BatchedEnv` implementation: `num_envs` independent copies of
    one `JaxEnv` stepped in parallel via `jax.vmap`, with per-slot auto-reset.

    Parameters
    ----------
    env : JaxEnv
        Single-instance environment to vectorise
    num_envs : int
        Number of parallel environments (`n_slots`)
    """

    def __init__(
        self,
        env: JaxEnv[ObsSpaceT, ActSpaceT, StateT, ConfigT],
        num_envs: int,
    ) -> None:
        self.env = env
        self.num_envs = num_envs

    @property
    def n_slots(self) -> int:
        """Number of parallel slots (= `num_envs`)."""
        return self.num_envs

    @property
    def name(self) -> str:
        """
        Inner environment's `name`. Used as the default key by `MultiVecEnv`.

        Picks up wrapper delegation, so `VecEnv(JitWrapper(BallEnv()))`
        still reports `"BallEnv"` rather than `"JitWrapper"`.

        Returns
        -------
        name : str
            The wrapped environment's `name`.
        """
        return self.env.name

    @property
    def config(self) -> ConfigT:
        """Single environment configuration."""
        return self.env.config

    def reset(self, rng: chex.PRNGKey) -> Tuple[jax.Array, StateT]:
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
        obs  : jax.Array
            Stacked first observations
        states : EnvState
            Batched environment states
        """
        rngs = jax.random.split(rng, self.num_envs)
        return jax.vmap(self.env.reset)(rngs)

    def step(
        self,
        state: StateT,
        actions: jax.Array,
    ) -> Tuple[jax.Array, StateT, jax.Array, jax.Array, Dict[str, Any]]:
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
        actions  : jax.Array
            One action per environment

        Returns
        -------
        obs  : jax.Array
            Observations after the step
        new_states : EnvState
            Updated batched states
        rewards  : jax.Array
            Per-environment rewards
        dones  : jax.Array
            Per-environment terminal flags
        infos : Dict[str, Any]
            Batched info dict
        """
        return jax.vmap(self._step_env)(state, actions)

    def _step_env(
        self,
        state: StateT,
        action: jax.Array,
    ) -> Tuple[jax.Array, StateT, jax.Array, jax.Array, Dict[str, Any]]:
        """
        Single-env step that auto-resets on episode end.

        When `done` is `True`, returns the observation from a fresh reset
        instead of the terminal observation.

        Parameters
        ----------
        state : EnvState
            Current environment state
        action : jax.Array
            Action to take in the environment

        Returns
        -------
        Tuple of `(obs, new_state, reward, done, info)`, where obs/new_state
        come from reset if done is `True`.
        """
        obs, new_state, reward, done, info = self.env.step(state, action)

        reset_rng, _ = jax.random.split(new_state.rng)
        reset_obs, reset_state = self.env.reset(reset_rng)

        final_obs = jnp.where(done, reset_obs, obs)
        final_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n), reset_state, new_state
        )

        return final_obs, final_state, reward, done, info

    def slot_state(self, state: StateT, slot_idx: int) -> StateT:
        """
        Extract the state pytree for a single slot from the batched state.

        Parameters
        ----------
        state : EnvState
            Batched environment state
        slot_idx : int
            Slot index in `[0, num_envs)`

        Returns
        -------
        single_state : EnvState
            Unbatched state pytree for the chosen slot.
        """
        return jax.tree.map(lambda x: x[slot_idx], state)

    def render_slot(self, state: StateT, slot_idx: int) -> np.ndarray:
        """
        Render a single environment from the batch.

        Parameters
        ----------
        state : EnvState
            Batched environment state
        slot_idx : int
            Slot index in `[0, num_envs)`

        Returns
        -------
        frame : np.ndarray
            uint8 RGB array of shape `(H, W, 3)`
        """
        return self.env.render(self.slot_state(state, slot_idx))

    def compile(self, cache_dir: Path | str | None = DEFAULT_CACHE_DIR) -> None:
        """
        Trigger XLA compilation by running dummy `reset` + `step`.

        Runs once with `done=False` (typical path) and once with `done=True`
        on the first slot to warm both branches of the auto-reset path.

        Parameters
        ----------
        cache_dir : Path | str | None (optional)
            XLA cache directory. Defaults to `<cwd>/.jax_cache`.
        """
        setup_cache(cache_dir)

        _key = jax.random.key(0)
        _, _state = self.reset(_key)
        _action_rngs = jax.random.split(_key, self.num_envs)
        _dummy_actions = jax.vmap(self.env.action_space.sample)(_action_rngs)

        self.step(_state, _dummy_actions)

        _forced_done = _state.done.at[0].set(jnp.bool_(True))
        _state_done = _state.__replace__(done=_forced_done)
        self.step(_state_done, _dummy_actions)

    @property
    def single_observation_space(self) -> ObsSpaceT:
        """Observation space of a single inner environment."""
        return self.env.observation_space

    @property
    def single_action_space(self) -> ActSpaceT:
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
