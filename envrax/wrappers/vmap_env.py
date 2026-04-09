from typing import Any, Dict, Tuple

import chex
import jax

from envrax.base import EnvParams, EnvState, JaxEnv
from envrax.spaces import Space


class VmapEnv:
    """
    Wraps any `JaxEnv` to operate over a batch of environments simultaneously.

    Requires no changes to the underlying env — vmap handles everything.
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
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """
        Reset all `num_envs` environments with independent random starts.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key (split into `num_envs` sub-keys).
        params : EnvParams
            Shared environment parameters.

        Returns
        -------
        obs  : chex.Array
            Stacked first observations, leading dim = `num_envs`.
        states : EnvState
            Batched environment states, leading dim = `num_envs`.
        """
        rngs = jax.random.split(rng, self.num_envs)
        return jax.vmap(self.env.reset, in_axes=(0, None))(rngs, params)

    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        actions: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance all environments by one step simultaneously.

        Uses `step_env` (which auto-resets on episode end) rather than
        `step`, so each environment independently resets when done.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key (split into `num_envs` sub-keys).
        state : EnvState
            Batched environment states, leading dim = `num_envs`.
        actions  : chex.Array
            int32[num_envs] — One action per environment.
        params : EnvParams
            Shared environment parameters.

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
        return jax.vmap(self.env.step_env, in_axes=(0, 0, 0, None))(
            rngs, state, actions, params
        )

    @property
    def observation_space(self) -> Space:
        """Observation space of the inner environment."""
        return self.env.observation_space

    @property
    def action_space(self) -> Space:
        """Action space of the inner environment."""
        return self.env.action_space

    def __repr__(self) -> str:
        return f"VmapEnv<{self.env!r}, num_envs={self.num_envs}>"
