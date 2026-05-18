from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import chex
import jax
import numpy as np

from envrax.spaces import Space


class BatchedEnv(ABC):
    """
    Base class for an env that produces `n_slots` independent agent results per step.

    Implementations choose their batching strategy — `jax.vmap` over a single
    `JaxEnv`, a composite multi-agent scene, threaded backends, or any other
    approach. All implementations expose the same shape contract:

      * observations `(n_slots, *obs_shape)`
      * actions      `(n_slots, *action_shape)`
      * rewards      `(n_slots,)`
      * dones        `(n_slots,)`

    State is an implementation-specific pytree carried opaquely by the caller.
    """

    n_slots: int

    @property
    def name(self) -> str:
        """
        Default key used by `MultiVecEnv` when keys aren't supplied.

        Subclasses should override with something meaningful (e.g. the
        wrapped env's class name). Falls back to this `BatchedEnv` subclass
        name when not overridden.

        Returns
        -------
        name : str
            Short identifier for this batched env.
        """
        return type(self).__name__

    @property
    @abstractmethod
    def single_observation_space(self) -> Space:
        """Observation space of a single slot (unbatched)."""

    @property
    @abstractmethod
    def single_action_space(self) -> Space:
        """Action space of a single slot (unbatched)."""

    @abstractmethod
    def reset(self, rng: chex.PRNGKey) -> Tuple[jax.Array, Any]:
        """
        Reset all `n_slots` independent agents.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs : jax.Array
            Initial observations, shape `(n_slots, *obs_shape)`
        state : Any
            Batched state pytree
        """

    @abstractmethod
    def step(
        self,
        state: Any,
        actions: jax.Array,
    ) -> Tuple[jax.Array, Any, jax.Array, jax.Array, Dict[str, Any]]:
        """
        Step all `n_slots` agents independently with per-slot auto-reset.

        Parameters
        ----------
        state : Any
            Batched state from a previous reset or step
        actions : jax.Array
            Actions, shape `(n_slots, *action_shape)`

        Returns
        -------
        obs : jax.Array
            Observations after the step, shape `(n_slots, *obs_shape)`
        new_state : Any
            Updated batched state
        reward : jax.Array
            Per-slot rewards, shape `(n_slots,)`
        done : jax.Array
            Per-slot terminal flags, shape `(n_slots,)`
        info : Dict[str, Any]
            Batched info dict
        """

    @abstractmethod
    def slot_state(self, state: Any, slot_idx: int) -> Any:
        """
        Extract the single-slot state pytree for one agent.

        Parameters
        ----------
        state : Any
            Batched state
        slot_idx : int
            Slot index in `[0, n_slots)`

        Returns
        -------
        single_state : Any
            Pytree of the same structure as `state` but with leading slot
            dimension removed.
        """

    @abstractmethod
    def render_slot(self, state: Any, slot_idx: int) -> np.ndarray:
        """
        Render a single slot from the batched state as an RGB frame.

        Parameters
        ----------
        state : Any
            Batched state
        slot_idx : int
            Slot index in `[0, n_slots)`

        Returns
        -------
        frame : np.ndarray
            uint8 RGB array of shape `(H, W, 3)`
        """

    @abstractmethod
    def compile(self, cache_dir: Path | str | None = None) -> None:
        """
        Trigger XLA compilation by running dummy `reset` + `step` calls.

        Implementations should also warm any conditional branches (e.g. the
        reset path of auto-reset logic) so the persistent cache covers them.

        Parameters
        ----------
        cache_dir : Path | str | None (optional)
            XLA cache directory. Implementations may default to a stable
            project-relative path.
        """
