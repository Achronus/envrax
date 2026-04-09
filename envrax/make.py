import pathlib
from typing import List, Tuple, Type

import jax
import jax.numpy as jnp

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.base import EnvParams, JaxEnv
from envrax.registry import _REGISTRY
from envrax.wrappers.base import Wrapper, _WrapperFactory
from envrax.wrappers.jit_wrapper import JitWrapper
from envrax.wrappers.vmap_env import VmapEnv


def make(
    name: str,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> Tuple[JaxEnv, EnvParams]:
    """
    Create a single `JaxEnv`, optionally with wrappers applied.

    Parameters
    ----------
    name : str
        Registered environment name (e.g. ``"atari/breakout-v0"``).
    params : EnvParams (optional)
        Environment parameters. Defaults to the registered default params.
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
    params : EnvParams
        Environment parameters used by the environment.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(f"Unknown environment: {name!r}. Available: {available}")

    env_class, default_params = _REGISTRY[name]
    resolved_params = params if params is not None else default_params
    env: JaxEnv = env_class()

    if wrappers:
        for w in wrappers:
            env = w(env)

    if jit_compile:
        env = JitWrapper(env, cache_dir=cache_dir)
        _key = jax.random.PRNGKey(0)
        _, _state = env.reset(_key, resolved_params)
        env.step(_key, _state, env.action_space.sample(_key), resolved_params)

    return env, resolved_params


def make_vec(
    name: str,
    n_envs: int,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> Tuple[VmapEnv, EnvParams]:
    """
    Create a `VmapEnv` with `n_envs` parallel environments.

    Parameters
    ----------
    name : str
        Registered environment name.
    n_envs : int
        Number of parallel environments.
    params : EnvParams (optional)
        Environment parameters. Defaults to the registered default params.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes applied innermost-first. Applied before vectorisation.
    jit_compile : bool (optional)
        Run one warm-up `reset` + `step` to eagerly trigger XLA compilation.
        Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.

    Returns
    -------
    vec_env : VmapEnv
        Vectorised environment.
    params : EnvParams
        Environment parameters used by the environment.
    """
    inner_env, resolved_params = make(
        name,
        params=params,
        wrappers=wrappers,
        jit_compile=False,
        cache_dir=None,
    )

    vec_env = VmapEnv(inner_env, n_envs)

    if jit_compile:
        setup_cache(cache_dir)
        _key = jax.random.PRNGKey(0)
        _, _states = vec_env.reset(_key, resolved_params)
        vec_env.step(_key, _states, jnp.zeros(n_envs, dtype=jnp.int32), resolved_params)

    return vec_env, resolved_params


def make_multi(
    names: List[str],
    *,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> List[Tuple[JaxEnv, EnvParams]]:
    """Create one `(JaxEnv, EnvParams)` tuple per entry in `names`."""
    return [
        make(
            name,
            params=params,
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
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> List[Tuple[VmapEnv, EnvParams]]:
    """Create one `(VmapEnv, EnvParams)` tuple per entry in `names`."""
    return [
        make_vec(
            name,
            n_envs,
            params=params,
            wrappers=wrappers,
            jit_compile=jit_compile,
            cache_dir=cache_dir,
        )
        for name in names
    ]
