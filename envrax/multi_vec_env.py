from collections import Counter
from math import prod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import chex
import jax
import numpy as np
from tqdm import tqdm

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.batched_env import BatchedEnv
from envrax.spaces import Space


def _auto_key(batched_envs: List[BatchedEnv]) -> Dict[str, BatchedEnv]:
    """
    Derive dict keys from each `BatchedEnv.name`, suffixing duplicates.

    A unique name is used bare (e.g. `"CartpoleBalanceEnv"`); names that
    appear more than once get a zero-indexed suffix
    (`"CartpoleBalanceEnv_0"`, `"CartpoleBalanceEnv_1"`, ...).

    Parameters
    ----------
    batched_envs : List[BatchedEnv]
        Envs to key.

    Returns
    -------
    envs_dict : Dict[str, BatchedEnv]
        Derived keys preserving input order.
    """
    counts = Counter(env.name for env in batched_envs)
    counters: Dict[str, int] = {}
    envs_dict: Dict[str, BatchedEnv] = {}

    for env in batched_envs:
        name = env.name
        if counts[name] == 1:
            key = name
        else:
            idx = counters.get(name, 0)
            key = f"{name}_{idx}"
            counters[name] = idx + 1

        envs_dict[key] = env

    return envs_dict


class MultiVecEnv:
    """
    JAX-native container for multiple `BatchedEnv` instances keyed by env name.

    State is a dict-of-pytrees (`Dict[str, chex.ArrayTree]`). The
    cross-env-type dispatch runs as a Python loop at `jax.jit` trace time,
    producing one XLA computation per call that dispatches one inner-kernel
    per env type with no per-call Python overhead between them.

    Accepts either a list (keys derived from each env's `name` via
    `_auto_key`) or a dict (used as-is for explicit control).

    Parameters
    ----------
    batched_envs : List[BatchedEnv] | Dict[str, BatchedEnv]
        Envs to wrap. When a list, keys are derived from `env.name` with
        suffixes on duplicates. When a dict, keys are used verbatim.
        Iteration order is preserved.
    """

    def __init__(
        self,
        batched_envs: List[BatchedEnv] | Dict[str, BatchedEnv],
    ) -> None:
        if not batched_envs:
            raise ValueError("MultiVecEnv requires at least one 'BatchedEnv'.")

        if isinstance(batched_envs, dict):
            envs_dict = dict(batched_envs)
        else:
            envs_dict = _auto_key(list(batched_envs))

        self._batched_envs: Dict[str, BatchedEnv] = envs_dict
        self._keys: List[str] = list(self._batched_envs.keys())
        self._jit_reset = jax.jit(self._reset_impl)
        self._jit_step = jax.jit(self._step_impl)

    @property
    def batched_envs(self) -> Dict[str, BatchedEnv]:
        """The inner `BatchedEnv` instances keyed by env name."""
        return self._batched_envs

    @property
    def env_keys(self) -> List[str]:
        """Ordered list of env-type keys."""
        return list(self._keys)

    @property
    def n_envs(self) -> int:
        """Number of distinct env types (= number of `BatchedEnv` instances)."""
        return len(self._batched_envs)

    @property
    def total_slots(self) -> int:
        """Total number of individual agent slots across all env types."""
        return sum(e.n_slots for e in self._batched_envs.values())

    @property
    def slots_per_env(self) -> Dict[str, int]:
        """Per-env-type slot counts."""
        return {k: e.n_slots for k, e in self._batched_envs.items()}

    @property
    def single_observation_spaces(self) -> Dict[str, Space]:
        """Per-env-type unbatched observation spaces."""
        return {k: e.single_observation_space for k, e in self._batched_envs.items()}

    @property
    def single_action_spaces(self) -> Dict[str, Space]:
        """Per-env-type unbatched action spaces."""
        return {k: e.single_action_space for k, e in self._batched_envs.items()}

    @property
    def single_observation_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Per-env-type unbatched observation shapes."""
        return {k: s.shape for k, s in self.single_observation_spaces.items()}

    @property
    def single_action_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Per-env-type unbatched action shapes."""
        return {k: s.shape for k, s in self.single_action_spaces.items()}

    @property
    def single_observation_sizes(self) -> Dict[str, int]:
        """Per-env-type flat unbatched observation element counts."""
        return {
            k: int(prod(s.shape)) for k, s in self.single_observation_spaces.items()
        }

    @property
    def single_action_sizes(self) -> Dict[str, int]:
        """Per-env-type flat unbatched action element counts."""
        return {k: int(prod(s.shape)) for k, s in self.single_action_spaces.items()}

    @property
    def single_observation_dtypes(self) -> Dict[str, Type]:
        """Per-env-type unbatched observation dtypes."""
        return {k: s.dtype for k, s in self.single_observation_spaces.items()}

    @property
    def single_action_dtypes(self) -> Dict[str, Type]:
        """Per-env-type unbatched action dtypes."""
        return {k: s.dtype for k, s in self.single_action_spaces.items()}

    def pad_dims(self) -> Tuple[int, int]:
        """
        Return `(max_action_size, max_observation_size)` across env types.

        Returns
        -------
        action : int
            Largest flat action size across all env types.
        observation : int
            Largest flat observation size across all env types.
        """
        return (
            max(self.single_action_sizes.values()),
            max(self.single_observation_sizes.values()),
        )

    def reset(
        self, rng: chex.PRNGKey
    ) -> Tuple[Dict[str, jax.Array], Dict[str, chex.ArrayTree]]:
        """
        Reset all env types with independent PRNG sub-keys.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs : Dict[str, jax.Array]
            Per-env-type batched observations.
        states : Dict[str, chex.ArrayTree]
            Per-env-type batched state pytrees.
        """
        return self._jit_reset(rng)

    def _reset_impl(
        self, rng: chex.PRNGKey
    ) -> Tuple[Dict[str, jax.Array], Dict[str, chex.ArrayTree]]:
        """
        Unjitted body of `reset`. Wrapped by `self._jit_reset` in `__init__`.

        Splits `rng` into one sub-key per env type and traces each inner
        env's `reset` into the same XLA computation at jit time.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs : Dict[str, jax.Array]
            Per-env-type batched observations.
        states : Dict[str, chex.ArrayTree]
            Per-env-type batched state pytrees.
        """
        keys = jax.random.split(rng, len(self._keys))
        obs: Dict[str, jax.Array] = {}
        states: Dict[str, chex.ArrayTree] = {}

        for i, key in enumerate(self._keys):
            o, s = self._batched_envs[key].reset(keys[i])
            obs[key] = o
            states[key] = s

        return obs, states

    def step(
        self,
        states: Dict[str, chex.ArrayTree],
        actions: Dict[str, jax.Array],
    ) -> Tuple[
        Dict[str, jax.Array],
        Dict[str, chex.ArrayTree],
        Dict[str, jax.Array],
        Dict[str, jax.Array],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Step all env types simultaneously.

        Parameters
        ----------
        states : Dict[str, chex.ArrayTree]
            Per-env-type batched states from a previous reset or step.
        actions : Dict[str, jax.Array]
            Per-env-type batched actions.

        Returns
        -------
        obs : Dict[str, jax.Array]
            Per-env-type batched observations after the step.
        new_states : Dict[str, chex.ArrayTree]
            Per-env-type updated batched states.
        rewards : Dict[str, jax.Array]
            Per-env-type batched rewards.
        dones : Dict[str, jax.Array]
            Per-env-type batched terminal flags.
        infos : Dict[str, Dict[str, Any]]
            Per-env-type batched info dicts.

        Raises
        ------
        key_mismatch : ValueError
            If `states` or `actions` keys do not match `env_keys`.
        """
        if set(states.keys()) != set(self._keys):
            raise ValueError(
                f"MultiVecEnv.step: `states` keys {sorted(states.keys())} "
                f"do not match env keys {sorted(self._keys)}."
            )

        if set(actions.keys()) != set(self._keys):
            raise ValueError(
                f"MultiVecEnv.step: `actions` keys {sorted(actions.keys())} "
                f"do not match env keys {sorted(self._keys)}."
            )

        return self._jit_step(states, actions)

    def _step_impl(
        self,
        states: Dict[str, chex.ArrayTree],
        actions: Dict[str, jax.Array],
    ) -> Tuple[
        Dict[str, jax.Array],
        Dict[str, chex.ArrayTree],
        Dict[str, jax.Array],
        Dict[str, jax.Array],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Unjitted body of `step`. Wrapped by `self._jit_step` in `__init__`.

        Traces each inner env's `step` into the same XLA computation at jit
        time — the Python loop over `self._keys` unrolls at tracing, not at
        runtime.

        Parameters
        ----------
        states : Dict[str, chex.ArrayTree]
            Per-env-type batched states from a previous reset or step.
        actions : Dict[str, jax.Array]
            Per-env-type batched actions.

        Returns
        -------
        obs : Dict[str, jax.Array]
            Per-env-type batched observations after the step.
        new_states : Dict[str, chex.ArrayTree]
            Per-env-type updated batched states.
        rewards : Dict[str, jax.Array]
            Per-env-type batched rewards.
        dones : Dict[str, jax.Array]
            Per-env-type batched terminal flags.
        infos : Dict[str, Dict[str, Any]]
            Per-env-type batched info dicts.
        """
        obs: Dict[str, jax.Array] = {}
        new_states: Dict[str, chex.ArrayTree] = {}
        rewards: Dict[str, jax.Array] = {}
        dones: Dict[str, jax.Array] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        for key in self._keys:
            o, s, r, d, info = self._batched_envs[key].step(states[key], actions[key])
            obs[key] = o
            new_states[key] = s
            rewards[key] = r
            dones[key] = d
            infos[key] = info

        return obs, new_states, rewards, dones, infos

    def slot_state(
        self, states: Dict[str, chex.ArrayTree], key: str, slot_idx: int
    ) -> chex.ArrayTree:
        """
        Extract the single-slot state pytree for one agent.

        Parameters
        ----------
        states : Dict[str, chex.ArrayTree]
            Per-env-type batched states.
        key : str
            Env-type key in `env_keys`.
        slot_idx : int
            Slot index in `[0, slots_per_env[key])`.

        Returns
        -------
        single_state : chex.ArrayTree
            Unbatched state pytree for the chosen slot.
        """
        return self._batched_envs[key].slot_state(states[key], slot_idx)

    def render_slot(
        self, states: Dict[str, chex.ArrayTree], key: str, slot_idx: int
    ) -> np.ndarray:
        """
        Render a single slot as an RGB frame.

        Parameters
        ----------
        states : Dict[str, chex.ArrayTree]
            Per-env-type batched states.
        key : str
            Env-type key in `env_keys`.
        slot_idx : int
            Slot index in `[0, slots_per_env[key])`.

        Returns
        -------
        frame : np.ndarray
            uint8 RGB array of shape `(H, W, 3)`.
        """
        return self._batched_envs[key].render_slot(states[key], slot_idx)

    def compile(
        self,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        *,
        progress: bool = True,
    ) -> None:
        """
        Trigger XLA compilation for all inner envs and warm the multi-step jit.

        Parameters
        ----------
        cache_dir : Path | str | None (optional)
            XLA cache directory. Defaults to `<cwd>/.jax_cache`.
        progress : bool (optional)
            Show a `tqdm` progress bar. Default is `True`.
        """
        setup_cache(cache_dir)

        it = (
            tqdm(self._batched_envs.items(), desc="Compiling batched envs", unit="env")
            if progress
            else self._batched_envs.items()
        )
        for _, env in it:
            env.compile(cache_dir=cache_dir)

        _rng = jax.random.key(0)
        _obs, _states = self.reset(_rng)
        _action_keys = jax.random.split(_rng, len(self._keys))
        _dummy_actions = {
            k: jax.vmap(self._batched_envs[k].single_action_space.sample)(
                jax.random.split(_action_keys[i], self._batched_envs[k].n_slots)
            )
            for i, k in enumerate(self._keys)
        }
        self.step(_states, _dummy_actions)

    def __len__(self) -> int:
        return self.n_envs

    def __repr__(self) -> str:
        group_info = ", ".join(
            f"{k}*{e.n_slots}" for k, e in self._batched_envs.items()
        )
        return f"MultiVecEnv({{{group_info}}}, total_slots={self.total_slots})"
