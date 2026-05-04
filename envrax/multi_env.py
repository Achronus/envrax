from collections import defaultdict
from typing import Any, Dict, List, Tuple

import chex
import jax
from tqdm import tqdm

from envrax.env import EnvState, JaxEnv
from envrax.spaces import Space
from envrax.wrappers.jit_wrapper import JitWrapper


def _build_class_groups(envs: List[JaxEnv]) -> Dict[str, List[int]]:
    """Group env indices by their class name."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, env in enumerate(envs):
        cls_name = type(env).__qualname__
        groups[cls_name].append(i)

    return dict(groups)


class MultiEnv:
    """
    Manages `M` heterogeneous `JaxEnv` instances as a single unit.
    Useful for holding `M` different `JaxEnv`s — with potentially different
    classes, configs, and shapes.

    Use `.class_groups` to identify which indices share a class for
    downstream batching of same-shape observations.

    Parameters
    ----------
    envs : List[JaxEnv]
        List of already-constructed environments.
    """

    def __init__(self, envs: List[JaxEnv]) -> None:
        if not envs:
            raise ValueError("MultiEnv requires at least one environment.")

        self._envs = envs
        self._class_groups = _build_class_groups(envs)

    @property
    def num_envs(self) -> int:
        """Number of environments (`M`)."""
        return len(self._envs)

    @property
    def envs(self) -> List[JaxEnv]:
        """The inner environment instances."""
        return self._envs

    @property
    def observation_spaces(self) -> List[Space]:
        """Per-env observation spaces."""
        return [env.observation_space for env in self._envs]

    @property
    def action_spaces(self) -> List[Space]:
        """Per-env action spaces."""
        return [env.action_space for env in self._envs]

    @property
    def class_groups(self) -> Dict[str, List[int]]:
        """
        Env class name → list of indices.

        Useful for downstream code that wants to batch operations across
        envs of the same class (e.g. stacking observations with matching
        shapes).
        """
        return self._class_groups

    def reset(self, rng: chex.PRNGKey) -> Tuple[List[chex.Array], List[EnvState]]:
        """
        Reset all `M` environments with independent PRNG keys.

        Splits `rng` into `M` sub-keys deterministically. Same master key
        produces the same per-env keys for full reproducibility.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        observations : List[chex.Array]
            Per-env initial observations
        states : List[EnvState]
            Per-env initial states
        """
        rngs = jax.random.split(rng, self.num_envs)
        obs_list: List[chex.Array] = []
        state_list: List[EnvState] = []

        for i, env in enumerate(self._envs):
            obs, state = env.reset(rngs[i])
            obs_list.append(obs)
            state_list.append(state)

        return obs_list, state_list

    def step(
        self,
        states: List[EnvState],
        actions: List[chex.Array],
    ) -> Tuple[
        List[chex.Array],
        List[EnvState],
        List[chex.Array],
        List[chex.Array],
        List[Dict[str, Any]],
    ]:
        """
        Step all `M` environments simultaneously.

        Parameters
        ----------
        states : List[EnvState]
            Per-env states from a previous reset or step
        actions : List[chex.Array]
            Per-env actions matching each env's action space

        Returns
        -------
        observations : List[chex.Array]
            Per-env observations after the step
        new_states : List[EnvState]
            Per-env updated states
        rewards : List[chex.Array]
            Per-env scalar rewards
        dones : List[chex.Array]
            Per-env terminal flags
        infos : List[Dict[str, Any]]
            Per-env info dicts

        Raises
        ------
        length_mismatch : ValueError
            If `len(states)` or `len(actions)` does not match `num_envs`.
        """
        if len(states) != self.num_envs or len(actions) != self.num_envs:
            raise ValueError(
                f"MultiEnv.step: expected {self.num_envs} states and actions, "
                f"got {len(states)} states and {len(actions)} actions."
            )

        results = [
            env.step(state, action)
            for env, state, action in zip(self._envs, states, actions)
        ]
        return (
            [r[0] for r in results],
            [r[1] for r in results],
            [r[2] for r in results],
            [r[3] for r in results],
            [r[4] for r in results],
        )

    def reset_at(self, idx: int, rng: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """
        Reset a single environment by index.

        Parameters
        ----------
        idx : int
            Index of the environment to reset
        rng : chex.PRNGKey
            JAX PRNG key for the reset

        Returns
        -------
        obs : chex.Array
            Initial observation
        state : EnvState
            Initial state
        """
        return self._envs[idx].reset(rng)

    def step_at(
        self,
        idx: int,
        state: EnvState,
        action: chex.Array,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step a single environment by index.

        Parameters
        ----------
        idx : int
            Index of the environment to step
        state : EnvState
            Current state of the environment
        action : chex.Array
            Action to take

        Returns
        -------
        obs : chex.Array
            Observation after the step
        new_state : EnvState
            Updated state
        reward : chex.Array
            Scalar reward
        done : chex.Array
            Terminal flag
        info : Dict[str, Any]
            Info dict
        """
        return self._envs[idx].step(state, action)

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
            (i, env) for i, env in enumerate(self._envs) if isinstance(env, JitWrapper)
        ]
        if not jit_envs:
            return

        it = tqdm(jit_envs, desc="Compiling envs", unit="env") if progress else jit_envs
        for _, env in it:
            env.compile()

    def __len__(self) -> int:
        return len(self._envs)

    def __repr__(self) -> str:
        env_info = ", ".join(type(e).__name__ for e in self._envs)
        return f"MultiEnv([{env_info}], num_envs={self.num_envs})"
