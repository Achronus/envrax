from collections import defaultdict
from typing import Any, Dict, List, Tuple

import chex
import jax
from tqdm import tqdm

from envrax.env import EnvState
from envrax.spaces import Space
from envrax.vec_env import VecEnv


def _build_class_groups(vec_envs: List[VecEnv]) -> Dict[str, List[int]]:
    """Group VecEnv indices by their inner env's class name."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, vec in enumerate(vec_envs):
        cls_name = type(vec.env).__qualname__
        groups[cls_name].append(i)

    return dict(groups)


class MultiVecEnv:
    """
    Manages `M` heterogeneous `VecEnv` instances as a single unit.
    Useful for holding `M` different `VecEnv`s — with potentially different
    classes, configs, and shapes.

    Use `.class_groups` to identify which indices share an inner env
    class for downstream batching.

    Parameters
    ----------
    vec_envs : List[VecEnv]
        List of already-constructed vectorised environments.
    """

    def __init__(self, vec_envs: List[VecEnv]) -> None:
        if not vec_envs:
            raise ValueError("MultiVecEnv requires at least one VecEnv.")

        self._vec_envs = vec_envs
        self._class_groups = _build_class_groups(vec_envs)

    @property
    def num_envs(self) -> int:
        """Number of VecEnv groups (`M`)."""
        return len(self._vec_envs)

    @property
    def total_envs(self) -> int:
        """Total number of individual environments across all groups."""
        return sum(v.num_envs for v in self._vec_envs)

    @property
    def vec_envs(self) -> List[VecEnv]:
        """The inner VecEnv instances."""
        return self._vec_envs

    @property
    def observation_spaces(self) -> List[Space]:
        """Per-group batched observation spaces."""
        return [v.observation_space for v in self._vec_envs]

    @property
    def action_spaces(self) -> List[Space]:
        """Per-group batched action spaces."""
        return [v.action_space for v in self._vec_envs]

    @property
    def single_observation_spaces(self) -> List[Space]:
        """Per-group unbatched observation spaces."""
        return [v.single_observation_space for v in self._vec_envs]

    @property
    def single_action_spaces(self) -> List[Space]:
        """Per-group unbatched action spaces."""
        return [v.single_action_space for v in self._vec_envs]

    @property
    def class_groups(self) -> Dict[str, List[int]]:
        """
        Inner env class name → list of VecEnv indices.

        Useful for downstream code that wants to batch operations across
        groups sharing the same inner env class.
        """
        return self._class_groups

    def reset(self, rng: chex.PRNGKey) -> Tuple[List[chex.Array], List[EnvState]]:
        """
        Reset all `M` vectorised environment groups with independent PRNG keys.

        Each `VecEnv` receives one sub-key and splits it internally across
        its own parallel copies.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        observations : List[chex.Array]
            Per-group batched observations
        states : List[EnvState]
            Per-group batched states
        """
        rngs = jax.random.split(rng, self.num_envs)
        obs_list: List[chex.Array] = []
        state_list: List[EnvState] = []

        for i, vec in enumerate(self._vec_envs):
            obs, state = vec.reset(rngs[i])
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
        Step all `M` vectorised environment groups simultaneously.

        Parameters
        ----------
        states : List[EnvState]
            Per-group batched states from a previous reset or step
        actions : List[chex.Array]
            Per-group batched actions

        Returns
        -------
        observations : List[chex.Array]
            Per-group batched observations after the step
        new_states : List[EnvState]
            Per-group updated batched states
        rewards : List[chex.Array]
            Per-group batched rewards
        dones : List[chex.Array]
            Per-group batched terminal flags
        infos : List[Dict[str, Any]]
            Per-group batched info dicts
        """
        results = [
            vec.step(state, action)
            for vec, state, action in zip(self._vec_envs, states, actions)
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
        Reset a single `VecEnv` group by index.

        Parameters
        ----------
        idx : int
            Index of the `VecEnv` group to reset
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            Batched initial observations for this group
        state : EnvState
            Batched initial state for this group
        """
        return self._vec_envs[idx].reset(rng)

    def step_at(
        self,
        idx: int,
        state: EnvState,
        action: chex.Array,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step a single `VecEnv` group by index.

        Parameters
        ----------
        idx : int
            Index of the `VecEnv` group to step
        state : EnvState
            Batched state for this group
        action : chex.Array
            Batched action for this group

        Returns
        -------
        obs : chex.Array
            Batched observations after the step
        new_state : EnvState
            Updated batched state
        reward : chex.Array
            Batched rewards
        done : chex.Array
            Batched terminal flags
        info : Dict[str, Any]
            Batched info dict
        """
        return self._vec_envs[idx].step(state, action)

    def compile(self, *, progress: bool = True) -> None:
        """
        Trigger XLA compilation for all inner `VecEnv` instances.

        Calls `compile()` on each `VecEnv`, which runs a dummy
        `reset` + `step` to populate the XLA cache.

        Parameters
        ----------
        progress : bool (optional)
            Show a `tqdm` progress bar. Default is `True`.
        """
        it = (
            tqdm(self._vec_envs, desc="Compiling vec envs", unit="env")
            if progress
            else self._vec_envs
        )
        for vec in it:
            vec.compile()

    def __len__(self) -> int:
        return len(self._vec_envs)

    def __repr__(self) -> str:
        group_info = ", ".join(
            f"{type(v.env).__name__}×{v.num_envs}" for v in self._vec_envs
        )
        return f"MultiVecEnv([{group_info}], total={self.total_envs})"
