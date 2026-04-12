from pathlib import Path
from typing import List, Tuple

from envrax._compile import DEFAULT_CACHE_DIR
from envrax.env import EnvConfig, JaxEnv
from envrax.multi_env import MultiEnv
from envrax.multi_vec_env import MultiVecEnv
from envrax.registry import _REGISTRY
from envrax.vec_env import VecEnv
from envrax.wrappers.base import WrapperType
from envrax.wrappers.jit_wrapper import JitWrapper


def make(
    name: str,
    *,
    config: EnvConfig | None = None,
    wrappers: List[WrapperType] | None = None,
    jit_compile: bool = True,
    pre_warm: bool = True,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> Tuple[JaxEnv, EnvConfig]:
    """
    Create a single `JaxEnv`, optionally with wrappers applied.

    Parameters
    ----------
    name : str
        Registered environment name
    config : EnvConfig (optional)
        Environment configuration. Defaults to the registered default config.
    wrappers : List[WrapperType] (optional)
        Wrapper classes or pre-configured factories applied innermost-first
        around the base env.
    jit_compile : bool (optional)
        Wrap the env in `JitWrapper`. Default is `True`.
    pre_warm : bool (optional)
        When `jit_compile=True`, run a dummy `reset` + `step` immediately
        to trigger XLA compilation. Set to `False` to defer compilation
        to the first real call or an explicit `compile()`. Default is `True`.
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
    unknown_env : ValueError
        If `name` is not registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown environment: {name!r}. Available: {available}")

    spec = _REGISTRY[name]
    resolved_config = config if config is not None else spec.default_config
    env: JaxEnv = spec.env_class(config=resolved_config)

    if wrappers:
        for w in wrappers:
            env = w(env)

    if jit_compile:
        env = JitWrapper(env, cache_dir=cache_dir, pre_warm=pre_warm)

    return env, resolved_config


def make_vec(
    name: str,
    n_envs: int,
    *,
    config: EnvConfig | None = None,
    wrappers: List[WrapperType] | None = None,
    jit_compile: bool = True,
    pre_warm: bool = True,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
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
    wrappers : List[WrapperType] (optional)
        Wrapper classes applied innermost-first. Applied before vectorisation.
    jit_compile : bool (optional)
        Enable the XLA compilation cache. Default is `True`.
    pre_warm : bool (optional)
        When `jit_compile=True`, run a dummy `reset` + `step` immediately.
        Set to `False` to defer to an explicit `vec_env.compile()` call.
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

    if jit_compile and pre_warm:
        vec_env.compile(cache_dir=cache_dir)

    return vec_env, resolved_config


def make_multi(
    names: List[str],
    *,
    config: EnvConfig | None = None,
    wrappers: List[WrapperType] | None = None,
    jit_compile: bool = True,
    pre_warm: bool = False,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> MultiEnv:
    """
    Create a `MultiEnv` managing M heterogeneous environments.

    By default, `pre_warm=False` so environments are JIT-wrapped but not
    compiled immediately. Call `multi_env.compile()` to trigger compilation
    as a separate setup phase.

    Parameters
    ----------
    names : List[str]
        Registered environment names
    config : EnvConfig (optional)
        Environment configuration applied to all envs.
        Defaults to each env's registered default.
    wrappers : List[WrapperType] (optional)
        Wrapper pipeline applied to each env
    jit_compile : bool (optional)
        Wrap each env in `JitWrapper`. Default is `True`.
    pre_warm : bool (optional)
        When `jit_compile=True`, compile each env immediately on creation.
        Default is `False` — call `multi_env.compile()` later instead.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache

    Returns
    -------
    multi_env : MultiEnv
        Manager holding all M environments
    """
    envs = []
    for name in names:
        env, _ = make(
            name,
            config=config,
            wrappers=wrappers,
            jit_compile=jit_compile,
            pre_warm=pre_warm,
            cache_dir=cache_dir,
        )
        envs.append(env)

    return MultiEnv(envs)


def make_multi_vec(
    names: List[str],
    n_envs: int,
    *,
    config: EnvConfig | None = None,
    wrappers: List[WrapperType] | None = None,
    jit_compile: bool = True,
    pre_warm: bool = False,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> MultiVecEnv:
    """
    Create a `MultiVecEnv` managing M heterogeneous vectorised environments.

    By default, `pre_warm=False` so VecEnv instances are created but not
    compiled immediately. Call `multi_vec_env.compile()` to trigger
    compilation as a separate setup phase.

    Parameters
    ----------
    names : List[str]
        Registered environment names
    n_envs : int
        Number of parallel copies per env
    config : EnvConfig (optional)
        Environment configuration applied to all envs.
        Defaults to each env's registered default.
    wrappers : List[WrapperType] (optional)
        Wrapper pipeline applied to each inner env before vectorisation
    jit_compile : bool (optional)
        Enable the XLA compilation cache. Default is `True`.
    pre_warm : bool (optional)
        When `jit_compile=True`, compile each VecEnv immediately.
        Default is `False` — call `multi_vec_env.compile()` later instead.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache

    Returns
    -------
    multi_vec_env : MultiVecEnv
        Manager holding all M vectorised environments
    """
    vec_envs = []
    for name in names:
        inner, _ = make(
            name,
            config=config,
            wrappers=wrappers,
            jit_compile=False,
            cache_dir=None,
        )
        vec = VecEnv(inner, n_envs)

        if jit_compile and pre_warm:
            vec.compile(cache_dir=cache_dir)

        vec_envs.append(vec)

    return MultiVecEnv(vec_envs)
