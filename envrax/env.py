from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Tuple, TypeVar

import chex

from envrax.spaces import Space

ObsSpaceT = TypeVar("ObsSpaceT", bound=Space)
ActSpaceT = TypeVar("ActSpaceT", bound=Space)
StateT = TypeVar("StateT", bound="EnvState")


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


class JaxEnv(ABC, Generic[ObsSpaceT, ActSpaceT, StateT]):
    """
    Base class for all JAX-native environments.

    Generic over the observation space, action space, and state types so
    that subclasses and wrappers get accurate type info without runtime
    casts. Subclasses pin all three:

        class BallEnv(JaxEnv[Box, Discrete, BallState]): ...

    Parameters
    ----------
    config : EnvConfig (optional)
        Static environment configuration. Defaults to `EnvConfig()`.
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config if config is not None else EnvConfig()

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
