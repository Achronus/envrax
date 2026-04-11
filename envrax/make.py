import pathlib
from typing import List, Tuple, Type

import jax
import jax.numpy as jnp

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.base import EnvConfig, JaxEnv
from envrax.registry import _REGISTRY
from envrax.vec_env import VecEnv
from envrax.wrappers.base import Wrapper, _WrapperFactory
from envrax.wrappers.jit_wrapper import JitWrapper


def make(
    name: str,
    *,
    config: EnvConfig | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> Tuple[JaxEnv, EnvConfig]:
    """
    Create a single `JaxEnv`, optionally with wrappers applied.

    Parameters
    ----------
    name : str
        Registered environment name
    config : EnvConfig (optional)
        Environment configuration. Defaults to the registered default config.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes or pre-configured factories applied innermost-first
        around the base env.
    jit_compile : bool (optional)
        Wrap the env in `JitWrapper` and run one warm-up `reset` + `step`
        to eagerly trigger XLA compilation. Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.
        Defaults to `~/.cache/envrax/xla_cache`. Pass `None` to disable.

    Returns
    -------
    env : JaxEnv
        Configured environment, wrapped in `JitWrapper` when `jit_compile=True`.
    config : EnvConfig
        Environment configuration used by the environment (also accessible
        via `env.config`).

    Raises
    ------
    ValueError
        If `name` is not registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(f"Unknown environment: {name!r}. Available: {available}")

    env_class, default_config = _REGISTRY[name]
    resolved_config = config if config is not None else default_config
    env: JaxEnv = env_class(config=resolved_config)

    if wrappers:
        for w in wrappers:
            env = w(env)

    if jit_compile:
        env = JitWrapper(env, cache_dir=cache_dir)
        _key = jax.random.key(0)
        _, _state = env.reset(_key)
        env.step(_state, env.action_space.sample(_key))

    return env, resolved_config


def make_vec(
    name: str,
    n_envs: int,
    *,
    config: EnvConfig | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> Tuple[VecEnv, EnvConfig]:
    """
    Create a `VecEnv` with `n_envs` parallel environments.

    Parameters
    ----------
    name : str
        Registered environment name
    n_envs : int
        Number of parallel environments
    config : EnvConfig (optional)
        Environment configuration. Defaults to the registered default config.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes applied innermost-first. Applied before vectorisation.
    jit_compile : bool (optional)
        Run one warm-up `reset` + `step` to eagerly trigger XLA compilation.
        Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.

    Returns
    -------
    vec_env : VecEnv
        Vectorised environment
    config : EnvConfig
        Environment configuration used by the environment (also accessible
        via `vec_env.config`).
    """
    inner_env, resolved_config = make(
        name,
        config=config,
        wrappers=wrappers,
        jit_compile=False,
        cache_dir=None,
    )

    vec_env = VecEnv(inner_env, n_envs)

    if jit_compile:
        setup_cache(cache_dir)
        _, _states = vec_env.reset(seed=0)
        vec_env.step(_states, jnp.zeros(n_envs, dtype=jnp.int32))

    return vec_env, resolved_config


def make_multi(
    names: List[str],
    *,
    config: EnvConfig | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> List[Tuple[JaxEnv, EnvConfig]]:
    """Create one `(JaxEnv, EnvConfig)` tuple per entry in `names`."""
    return [
        make(
            name,
            config=config,
            wrappers=wrappers,
            jit_compile=jit_compile,
            cache_dir=cache_dir,
        )
        for name in names
    ]


def make_multi_vec(
    names: List[str],
    n_envs: int,
    *,
    config: EnvConfig | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> List[Tuple[VecEnv, EnvConfig]]:
    """Create one `(VecEnv, EnvConfig)` tuple per entry in `names`."""
    return [
        make_vec(
            name,
            n_envs,
            config=config,
            wrappers=wrappers,
            jit_compile=jit_compile,
            cache_dir=cache_dir,
        )
        for name in names
    ]
