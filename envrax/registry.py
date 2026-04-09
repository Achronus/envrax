from typing import Dict, List, Tuple, Type

from envrax.base import EnvParams, JaxEnv

_REGISTRY: Dict[str, Tuple[Type[JaxEnv], EnvParams]] = {}


def register(name: str, env_class: Type[JaxEnv], default_params: EnvParams) -> None:
    """
    Register an environment class under a given name.

    Called on package import by downstream packages
    to make their environments available via `envrax.make()`.

    Parameters
    ----------
    name : str
        Unique environment name (e.g. `"atari/breakout-v0"`).
    env_class : Type[JaxEnv]
        The environment class to register.
    default_params : EnvParams
        Default parameters for this environment.

    Raises
    ------
    env_exists : ValueError
        If `name` is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(
            f"Environment {name!r} is already registered. "
            "Use a unique name or unregister the existing entry first."
        )
    _REGISTRY[name] = (env_class, default_params)


def make_env(name: str, **param_overrides) -> Tuple[JaxEnv, EnvParams]:
    """
    Create a bare environment instance by name (no JIT, no wrappers).

    For JIT compilation and wrapper support use `envrax.make()` instead.

    Parameters
    ----------
    name : str
        Registered environment name (e.g. `"atari/breakout-v0"`).
    **param_overrides
        Keyword arguments forwarded to `EnvParams.__replace__()` to override
        individual default parameters.

    Returns
    -------
    env : JaxEnv
        Instantiated environment.
    params : EnvParams
        Resolved parameters (defaults merged with any overrides).

    Raises
    ------
    env_exists : ValueError
        If `name` is not registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(f"Unknown environment: {name!r}. Available: {available}")
    env_class, default_params = _REGISTRY[name]
    params = (
        default_params.__replace__(**param_overrides)
        if param_overrides
        else default_params
    )
    return env_class(), params


def registered_names() -> List[str]:
    """Return a sorted list of all registered environment names."""
    return sorted(_REGISTRY)
