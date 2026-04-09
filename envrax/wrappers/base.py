from abc import abstractmethod
from typing import Any, Dict, Self, Tuple, overload

import chex

from envrax.base import EnvParams, EnvState, JaxEnv
from envrax.spaces import Box, Discrete


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


class Wrapper(JaxEnv):
    """
    Abstract base class for JaxEnv wrappers.

    Subclasses must implement `reset` and `step` with the JaxEnv API
    signatures: `reset(rng, params)` and `step(rng, state, action, params)`.

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

    def __init__(self, env: JaxEnv) -> None:
        self._env = env

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self._env!r}>"

    @abstractmethod
    def reset(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset the environment and return the initial observation and state."""
        raise NotImplementedError()

    @abstractmethod
    def step(
        self,
        rng: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """Advance the environment by one step."""
        raise NotImplementedError()

    @property
    def unwrapped(self) -> JaxEnv:
        """Return the innermost `JaxEnv` by delegating through the wrapper chain."""
        return self._env.unwrapped if isinstance(self._env, Wrapper) else self._env

    @property
    def observation_space(self) -> Box:
        """Observation space of the inner environment."""
        return self._env.observation_space

    @property
    def action_space(self) -> Discrete:
        """Action space of the inner environment."""
        return self._env.action_space
