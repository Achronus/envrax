from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Tuple, Type, TypeVar

import chex
import numpy as np

from envrax.spaces import Space
from envrax.utils import resolve_generic_arg

ObsSpaceT = TypeVar("ObsSpaceT", bound=Space)
ActSpaceT = TypeVar("ActSpaceT", bound=Space)
StateT = TypeVar("StateT", bound="EnvState")
ConfigT = TypeVar("ConfigT", bound="EnvConfig")


@chex.dataclass
class EnvState:
    """
    Base environment state. Every environment extends this with its own fields.

    All fields must be JAX arrays or Python scalars (for static shape info).
    No Python objects, no lists, no dicts.

    Parameters
    ----------
    rng : chex.PRNGKey
        JAX PRNG key
    step : chex.Array
        Current timestep within the episode
    done : chex.Array
        bool scalar — episode termination flag
    """

    rng: chex.PRNGKey
    step: chex.Array
    done: chex.Array


@chex.dataclass
class EnvConfig:
    """
    Static environment configuration. Set once at construction, never changed.
    Controls things like max steps, reward scaling, difficulty, etc.

    Parameters
    ----------
    max_steps : int
        Maximum number of steps per episode. Default is 1000.
    """

    max_steps: int = 1000


class JaxEnv(ABC, Generic[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
    """
    Base class for all JAX-native environments.

    Generic over the observation space, action space, state, and config
    types so that subclasses and wrappers get accurate type info without
    runtime casts. Subclasses pin all four:

        class BallEnv(JaxEnv[Box, Discrete, BallState, BallConfig]): ...

    Parameters
    ----------
    config : ConfigT (optional)
        Static environment configuration. Defaults to `ConfigT()`.
    """

    config: ConfigT

    def __init__(self, config: ConfigT | None = None) -> None:
        if config is None:
            config_cls = self._resolve_config_cls()
            config = config_cls()

        self.config = config  # type: ignore

    @property
    @abstractmethod
    def observation_space(self) -> ObsSpaceT:
        """Returns the observation space."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> ActSpaceT:
        """Returns the action space."""
        ...

    @abstractmethod
    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
        """
        Set the environment to a starting state.

        Implementations should split `rng` so one half is consumed for
        initialisation and the other half is stored on the returned state's
        `rng` field for `step` to use.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            Initial observation
        state : StateT
            Initial environment state with `rng` embedded
        """
        ...

    @abstractmethod
    def step(
        self,
        state: StateT,
        action: chex.Array,
    ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Take an action through the environment.

        Implementations should split `state.rng` for any per-step randomness
        and store the remaining key on `new_state.rng` so randomness threads
        through the episode.

        Parameters
        ----------
        state : StateT
            Current environment state
        action : chex.Array
            Action to take in the environment

        Returns
        -------
        obs : chex.Array
            Observation after the step
        new_state : StateT
            Updated environment state
        reward : chex.Array
            Scalar reward
        done : chex.Array
            bool scalar — `True` when the episode has ended, `False` otherwise
        info : Dict[str, Any]
            Auxiliary diagnostic information
        """
        ...

    def render(self, state: StateT) -> np.ndarray:
        """
        Render the environment state as an RGB frame.

        Parameters
        ----------
        state : StateT
            Current environment state to render

        Returns
        -------
        frame : np.ndarray
            uint8 RGB array of shape `(H, W, 3)`
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement render(). "
            "Override render(state) to return a uint8 (H, W, 3) RGB frame."
        )

    @classmethod
    def _resolve_config_cls(cls) -> Type:
        """
        Return the concrete `EnvConfig` subclass pinned via `JaxEnv[..., ConfigT]`.

        Returns
        -------
        config_cls : Type
            The class pinned to `ConfigT` for this subclass
        """
        return resolve_generic_arg(cls, JaxEnv, position=3)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
