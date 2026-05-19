from collections import Counter
from math import prod
from typing import Any, Dict, List, Tuple, Type

import chex
import jax
from tqdm import tqdm

from envrax.env import EnvState, JaxEnv
from envrax.spaces import Space
from envrax.wrappers.jit_wrapper import JitWrapper


def _auto_key(envs: List[JaxEnv]) -> Dict[str, JaxEnv]:
    """
    Derive dict keys from each `JaxEnv.name`, suffixing duplicates.

    A unique name is used bare (e.g. `"BallEnv"`); names that appear more
    than once get a zero-indexed suffix (`"BallEnv_0"`, `"BallEnv_1"`, ...).

    Parameters
    ----------
    envs : List[JaxEnv]
        Envs to key.

    Returns
    -------
    envs_dict : Dict[str, JaxEnv]
        Derived keys preserving input order.
    """
    counts = Counter(env.name for env in envs)
    counters: Dict[str, int] = {}
    envs_dict: Dict[str, JaxEnv] = {}

    for env in envs:
        name = env.name
        if counts[name] == 1:
            key = name
        else:
            idx = counters.get(name, 0)
            key = f"{name}_{idx}"
            counters[name] = idx + 1

        envs_dict[key] = env

    return envs_dict


class MultiEnv:
    """
    Container for multiple `JaxEnv` instances keyed by env name.

    Holds one inner env per key and dispatches `reset`/`step` via a Python
    loop. No outer `jax.jit` boundary is added — each inner env keeps its
    own compile cycle (typically via `JitWrapper`). Use `MultiVecEnv` if
    you need a single jitted dispatch over batched envs.

    Accepts either a list (keys derived from each env's `name` via
    `_auto_key`) or a dict (used as-is for explicit control).

    Parameters
    ----------
    envs : List[JaxEnv] | Dict[str, JaxEnv]
        Envs to wrap. When a list, keys are derived from `env.name` with
        suffixes on duplicates. When a dict, keys are used verbatim.
        Iteration order is preserved.
    """

    def __init__(
        self,
        envs: List[JaxEnv] | Dict[str, JaxEnv],
    ) -> None:
        if not envs:
            raise ValueError("MultiEnv requires at least one environment.")

        if isinstance(envs, dict):
            envs_dict = dict(envs)
        else:
            envs_dict = _auto_key(list(envs))

        self._envs: Dict[str, JaxEnv] = envs_dict
        self._keys: List[str] = list(self._envs.keys())

    @property
    def envs(self) -> Dict[str, JaxEnv]:
        """The inner `JaxEnv` instances keyed by env name."""
        return self._envs

    @property
    def env_keys(self) -> List[str]:
        """Ordered list of env-type keys."""
        return list(self._keys)

    @property
    def n_envs(self) -> int:
        """Number of environments (`M`)."""
        return len(self._envs)

    @property
    def observation_spaces(self) -> Dict[str, Space]:
        """Per-env observation spaces."""
        return {k: e.observation_space for k, e in self._envs.items()}

    @property
    def action_spaces(self) -> Dict[str, Space]:
        """Per-env action spaces."""
        return {k: e.action_space for k, e in self._envs.items()}

    @property
    def observation_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Per-env observation shapes."""
        return {k: s.shape for k, s in self.observation_spaces.items()}

    @property
    def action_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Per-env action shapes."""
        return {k: s.shape for k, s in self.action_spaces.items()}

    @property
    def observation_sizes(self) -> Dict[str, int]:
        """Per-env flat observation element counts (`prod(shape)`)."""
        return {k: int(prod(s.shape)) for k, s in self.observation_spaces.items()}

    @property
    def action_sizes(self) -> Dict[str, int]:
        """Per-env flat action element counts (`prod(shape)`)."""
        return {k: int(prod(s.shape)) for k, s in self.action_spaces.items()}

    @property
    def observation_dtypes(self) -> Dict[str, Type]:
        """Per-env observation dtypes."""
        return {k: s.dtype for k, s in self.observation_spaces.items()}

    @property
    def action_dtypes(self) -> Dict[str, Type]:
        """Per-env action dtypes."""
        return {k: s.dtype for k, s in self.action_spaces.items()}

    def pad_dims(self) -> Tuple[int, int]:
        """
        Return `(max_action_size, max_observation_size)` across envs.

        Returns
        -------
        action : int
            Largest flat action size.
        observation : int
            Largest flat observation size.
        """
        return (
            max(self.action_sizes.values()),
            max(self.observation_sizes.values()),
        )

    def reset(
        self, rng: chex.PRNGKey
    ) -> Tuple[Dict[str, jax.Array], Dict[str, EnvState]]:
        """
        Reset all environments with independent PRNG sub-keys.

        Splits `rng` deterministically. Same master key produces the same
        per-env keys for full reproducibility.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs : Dict[str, jax.Array]
            Per-env initial observations.
        states : Dict[str, EnvState]
            Per-env initial states.
        """
        keys = jax.random.split(rng, len(self._keys))
        obs: Dict[str, jax.Array] = {}
        states: Dict[str, EnvState] = {}

        for i, key in enumerate(self._keys):
            o, s = self._envs[key].reset(keys[i])
            obs[key] = o
            states[key] = s

        return obs, states

    def step(
        self,
        states: Dict[str, EnvState],
        actions: Dict[str, jax.Array],
    ) -> Tuple[
        Dict[str, jax.Array],
        Dict[str, EnvState],
        Dict[str, jax.Array],
        Dict[str, jax.Array],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Step all environments simultaneously.

        Parameters
        ----------
        states : Dict[str, EnvState]
            Per-env states from a previous reset or step.
        actions : Dict[str, jax.Array]
            Per-env actions matching each env's action space.

        Returns
        -------
        obs : Dict[str, jax.Array]
            Per-env observations after the step.
        new_states : Dict[str, EnvState]
            Per-env updated states.
        rewards : Dict[str, jax.Array]
            Per-env scalar rewards.
        dones : Dict[str, jax.Array]
            Per-env terminal flags.
        infos : Dict[str, Dict[str, Any]]
            Per-env info dicts.

        Raises
        ------
        key_mismatch : ValueError
            If `states` or `actions` keys do not match `env_keys`.
        """
        if set(states.keys()) != set(self._keys):
            raise ValueError(
                f"MultiEnv.step: `states` keys {sorted(states.keys())} "
                f"do not match env keys {sorted(self._keys)}."
            )

        if set(actions.keys()) != set(self._keys):
            raise ValueError(
                f"MultiEnv.step: `actions` keys {sorted(actions.keys())} "
                f"do not match env keys {sorted(self._keys)}."
            )

        obs: Dict[str, jax.Array] = {}
        new_states: Dict[str, EnvState] = {}
        rewards: Dict[str, jax.Array] = {}
        dones: Dict[str, jax.Array] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        for key in self._keys:
            o, s, r, d, info = self._envs[key].step(states[key], actions[key])
            obs[key] = o
            new_states[key] = s
            rewards[key] = r
            dones[key] = d
            infos[key] = info

        return obs, new_states, rewards, dones, infos

    def compile(self, *, progress: bool = True) -> None:
        """
        Trigger XLA compilation for all JIT-wrapped environments.

        Calls `compile()` on each inner env that is a `JitWrapper`.
        Environments without JIT wrapping are silently skipped.

        Parameters
        ----------
        progress : bool (optional)
            Show a `tqdm` progress bar. Default is `True`.
        """
        jit_envs = [
            (k, env) for k, env in self._envs.items() if isinstance(env, JitWrapper)
        ]
        if not jit_envs:
            return

        it = (
            tqdm(jit_envs, desc="Compiling envs", unit="env") if progress else jit_envs
        )
        for _, env in it:
            env.compile()

    def __len__(self) -> int:
        return self.n_envs

    def __repr__(self) -> str:
        group_info = ", ".join(f"{k}={type(e).__name__}" for k, e in self._envs.items())
        return f"MultiEnv({{{group_info}}}, n_envs={self.n_envs})"
