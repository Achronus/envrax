from typing import Dict, List, Type

from envrax.env import EnvConfig, JaxEnv
from envrax.suite import EnvSpec, EnvSuite

_REGISTRY: Dict[str, EnvSpec] = {}


def register(
    name: str,
    env_class: Type[JaxEnv],
    default_config: EnvConfig,
    *,
    suite: str = "",
) -> None:
    """
    Register an environment class under a given name.

    Makes the environment available via `envrax.make()` methods.

    Parameters
    ----------
    name : str
        Unique environment name (e.g. `"mjx/cartpole-v0"`).
    env_class : Type[JaxEnv]
        The environment class to register.
    default_config : EnvConfig
        Default configuration for this environment.
    suite : str (optional)
        Suite category tag for introspection (e.g. `"MuJoCo Playground"`).

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

    _REGISTRY[name] = EnvSpec(
        name=name,
        env_class=env_class,
        default_config=default_config,
        suite=suite,
    )


def register_suite(suite: EnvSuite, *, version: str | None = None) -> None:
    """
    Register every environment in an `EnvSuite` in one shot.

    Parameters
    ----------
    suite : EnvSuite
        The suite whose environments should be registered.
    version : str (optional)
        Override the suite's default version when computing canonical IDs.

    Raises
    ------
    env_exists : ValueError
        If any resulting canonical ID is already registered.
    """
    for spec in suite.specs:
        canonical = suite.get_name(spec.name, version)

        if canonical in _REGISTRY:
            raise ValueError(
                f"Environment {canonical!r} is already registered. "
                "Use a unique name or unregister the existing entry first."
            )

        _REGISTRY[canonical] = EnvSpec(
            name=canonical,
            env_class=spec.env_class,
            default_config=spec.default_config,
            suite=suite.category,
        )


def get_spec(name: str) -> EnvSpec:
    """
    Return the full `EnvSpec` for a registered environment.

    Parameters
    ----------
    name : str
        Registered environment name.

    Returns
    -------
    spec : EnvSpec
        The registered specification.

    Raises
    ------
    unknown_env : ValueError
        If `name` is not registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown environment: {name!r}. Available: {available}")

    return _REGISTRY[name]


def registered_names() -> List[str]:
    """
    Get a sorted list of all registered environments.

    Returns
    -------
    envs : List[str]
        List of environment canonical IDs stored in the registry.
    """
    return sorted(_REGISTRY.keys())
