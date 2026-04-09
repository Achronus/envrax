from typing import Any, Dict, Tuple

import chex
import jax

from envrax.base import EnvConfig, EnvState, JaxEnv
from envrax.spaces import Space, batch_space


class VecEnv:
    """
    Wraps any `JaxEnv` to operate over a batch of environments simultaneously.

    Requires no changes to the underlying env — `vmap` handles everything.
    Calling `reset` splits the PRNG key into `num_envs` sub-keys so that
    each environment starts from a distinct random state.

    Parameters
    ----------
    env : JaxEnv
        Single-instance environment to vectorise.
    num_envs : int
        Number of parallel environments.
    """

    def __init__(self, env: JaxEnv, num_envs: int) -> None:
        self.env = env
        self.num_envs = num_envs

    def reset(
        self,
        rng: chex.PRNGKey,
        config: EnvConfig,
    ) -> Tuple[chex.Array, EnvState]:
        """
        Reset all `num_envs` environments with independent random starts.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key
        config : EnvConfig
            Shared environment configuration

        Returns
        -------
        obs  : chex.Array
            Stacked first observations, leading dim = `num_envs`
        states : EnvState
            Batched environment states, leading dim = `num_envs`
        """
        rngs = jax.random.split(rng, self.num_envs)
        return jax.vmap(self.env.reset, in_axes=(0, None))(rngs, config)

    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        actions: chex.Array,
        config: EnvConfig,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance all environments by one step simultaneously.

        Environments independently auto-reset on episode when done.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key
        state : EnvState
            Batched environment states, leading dim = `num_envs`
        actions  : chex.Array
            `int32[num_envs]` — One action per environment
        config : EnvConfig
            Shared environment settings

        Returns
        -------
        obs  : chex.Array
            Observations after the step, leading dim = `num_envs`.
        new_states : EnvState
            Updated batched states.
        rewards  : chex.Array
            float32[num_envs] — Per-environment rewards.
        dones  : chex.Array
            bool[num_envs] — Per-environment terminal flags.
        infos : dict
            Batched info dict; each value has a leading `num_envs` dimension.
        """
        rngs = jax.random.split(rng, self.num_envs)
        return jax.vmap(self._step_env, in_axes=(0, 0, 0, None))(
            rngs,
            state,
            actions,
            config,
        )

    def _step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        config: EnvConfig,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Wraps `step()` to auto-reset on episode end.

        When done is `True`, returns the observation from a fresh reset
        instead of the terminal observation. Override only if custom
        reset behaviour is needed.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : EnvState
            Current environment state.
        action : chex.Array
            Action to take.
        config : EnvConfig
            Static environment configuration.

        Returns
        -------
        Tuple of `(obs, new_state, reward, done, info)`, where obs/new_state
        come from reset if done is `True`.
        """
        obs, new_state, reward, done, info = self.env.step(rng, state, action, config)

        reset_rng, _ = jax.random.split(rng)
        reset_obs, reset_state = self.env.reset(reset_rng, config)

        final_obs = jax.lax.cond(done, lambda: reset_obs, lambda: obs)
        final_state = jax.lax.cond(done, lambda: reset_state, lambda: new_state)

        return final_obs, final_state, reward, done, info

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
        """Batched observation space with a leading ``num_envs`` dimension."""
        return batch_space(self.env.observation_space, self.num_envs)

    @property
    def action_space(self) -> Space:
        """Batched action space with a leading ``num_envs`` dimension."""
        return batch_space(self.env.action_space, self.num_envs)

    def __repr__(self) -> str:
        return f"VmapEnv<{self.env!r}, num_envs={self.num_envs}>"
