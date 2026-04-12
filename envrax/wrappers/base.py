from abc import abstractmethod
from typing import Any, Dict, Generic, Self, Tuple, TypeVar, overload

import chex

from envrax.env import ActSpaceT, EnvState, JaxEnv, ObsSpaceT, StateT

InnerStateT = TypeVar("InnerStateT", bound=EnvState, default=StateT)


class _WrapperFactory:
    """
    Deferred wrapper returned by `Wrapper.__new__` when called without an `env`.

    Calling the factory with an `env` creates the intended wrapper with the
    pre-bound keyword arguments.

    Parameters
    ----------
    cls : type
        Concrete `Wrapper` subclass to instantiate.
    **kwargs
        Keyword arguments forwarded to `cls.__init__` when the factory is called.
    """

    __slots__ = ("_cls", "_kwargs")

    def __init__(self, cls: type, **kwargs) -> None:
        self._cls = cls
        self._kwargs = kwargs

    def __call__(self, env: JaxEnv) -> "Wrapper":
        """
        Wrap `env` using the stored class and keyword arguments.

        Parameters
        ----------
        env : JaxEnv
            Environment to wrap.

        Returns
        -------
        wrapper : Wrapper
            Configured wrapper instance.
        """
        return self._cls(env, **self._kwargs)


class Wrapper(
    JaxEnv[ObsSpaceT, ActSpaceT, StateT],
    Generic[ObsSpaceT, ActSpaceT, StateT, InnerStateT],
):
    """
    Abstract base class for JaxEnv wrappers.

    Uses generics over the inner env's observation space, action space, outer
    state type (`StateT`), and inner state type (`InnerStateT`). The
    `InnerStateT` parameter defaults to `StateT`, so wrappers
    that preserve the inner state type only need to declare three params.

    **Pass-through wrappers** preserve the inner env's state type. They
    declare three TypeVars and let `InnerStateT` default to `StateT`:

        class ClipReward(Wrapper[ObsSpaceT, ActSpaceT, StateT]): ...

    **Stateful wrappers** introduce their own outer state type that wraps
    the inner state. They declare all four TypeVars, pinning `StateT` to
    their wrapper-specific class and leaving `InnerStateT` parametric:

        class FrameStackObservation(Wrapper[Box, ActSpaceT, FrameStackState, InnerStateT]):
            def __init__(self, env: JaxEnv[Box, ActSpaceT, InnerStateT], ...) -> None: ...

    The `observation_space` and `action_space` properties delegate to the
    inner environment by default and may be overridden when the wrapper
    changes the observation shape or action set.

    Parameterised wrappers support a **factory mode**: calling the class
    without an `env` (using only keyword arguments) returns a
    `_WrapperFactory` rather than a live wrapper.

    Parameters
    ----------
    env : JaxEnv
        Inner environment to wrap.
    """

    _env: JaxEnv[ObsSpaceT, ActSpaceT, InnerStateT]

    @overload
    def __new__(cls, env: None = ..., **kwargs) -> "_WrapperFactory": ...

    @overload
    def __new__(cls, env: JaxEnv, **kwargs) -> Self: ...

    def __new__(cls, env=None, **kwargs):
        if env is None:
            factory = object.__new__(_WrapperFactory)
            _WrapperFactory.__init__(factory, cls, **kwargs)
            return factory
        return super().__new__(cls)

    def __init__(self, env: JaxEnv[ObsSpaceT, ActSpaceT, InnerStateT]) -> None:
        super().__init__(env.config)
        self._env = env

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self._env!r}>"

    @abstractmethod
    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
        """Reset the environment and return the initial observation and state."""
        raise NotImplementedError()

    @abstractmethod
    def step(
        self,
        state: StateT,
        action: chex.Array,
    ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
        """Advance the environment by one step."""
        raise NotImplementedError()

    @property
    def unwrapped(self) -> JaxEnv:
        """Return the innermost `JaxEnv` by delegating through the wrapper chain."""
        return self._env.unwrapped if isinstance(self._env, Wrapper) else self._env

    @property
    def observation_space(self) -> ObsSpaceT:
        """Observation space of the inner environment."""
        return self._env.observation_space

    @property
    def action_space(self) -> ActSpaceT:
        """Action space of the inner environment."""
        return self._env.action_space


type WrapperType = type[Wrapper] | _WrapperFactory
"""
A wrapper supplied to factory functions like `make()`.

Either a `Wrapper` subclass (applied directly to an env) or a deferred
factory returned by calling a parameterised wrapper class without an env
(e.g. `ResizeObservation(h=84, w=84)`).
"""
