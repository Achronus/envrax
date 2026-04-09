from typing import Dict, List, Tuple, Type

from envrax.base import EnvConfig, JaxEnv

_REGISTRY: Dict[str, Tuple[Type[JaxEnv], EnvConfig]] = {}


def register(name: str, env_class: Type[JaxEnv], default_config: EnvConfig) -> None:
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
    default_config : EnvConfig
        Default configuration for this environment.

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
    _REGISTRY[name] = (env_class, default_config)


def make_env(name: str, **config_overrides) -> Tuple[JaxEnv, EnvConfig]:
    """
    Create a bare environment instance by name (no JIT, no wrappers).

    For JIT compilation and wrapper support use `envrax.make()` instead.

    Parameters
    ----------
    name : str
        Registered environment name (e.g. `"atari/breakout-v0"`).
    **config_overrides
        Keyword arguments forwarded to `EnvConfig.__replace__()` to override
        individual default config values.

    Returns
    -------
    env : JaxEnv
        Instantiated environment.
    config : EnvConfig
        Resolved configuration (defaults merged with any overrides).

    Raises
    ------
    env_exists : ValueError
        If `name` is not registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(f"Unknown environment: {name!r}. Available: {available}")
    env_class, default_config = _REGISTRY[name]
    config = (
        default_config.__replace__(**config_overrides)
        if config_overrides
        else default_config
    )
    return env_class(), config


def registered_names() -> List[str]:
    """Return a sorted list of all registered environment names."""
    return sorted(_REGISTRY)
