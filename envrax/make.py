from pathlib import Path
from typing import Dict, List

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.batched_env import BatchedEnv
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
) -> JaxEnv:
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
        Defaults to `<cwd>/.jax_cache` (override with the
        `ENVRAX_CACHE_DIR` environment variable). Pass `None` to disable.

    Returns
    -------
    env : JaxEnv
        Configured environment, wrapped in `JitWrapper` when `jit_compile=True`.

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

    return env


def make_vec(
    name: str,
    n_envs: int,
    *,
    config: EnvConfig | None = None,
    wrappers: List[WrapperType] | None = None,
    jit_compile: bool = True,
    pre_warm: bool = True,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> VecEnv:
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
        Vectorised environment.
    """
    if jit_compile:
        setup_cache(cache_dir)

    inner_env = make(
        name,
        config=config,
        wrappers=wrappers,
        jit_compile=False,
        cache_dir=None,
    )

    vec_env = VecEnv(inner_env, n_envs)

    if jit_compile and pre_warm:
        vec_env.compile(cache_dir=cache_dir)

    return vec_env


def make_multi(
    envs: List[JaxEnv] | Dict[str, JaxEnv],
    *,
    pre_warm: bool = False,
) -> MultiEnv:
    """
    Wrap `JaxEnv` instances into a `MultiEnv`.

    Parameters
    ----------
    envs : List[JaxEnv] | Dict[str, JaxEnv]
        Envs to wrap. List form: keys derived from `env.name` with suffixes
        on duplicates. Dict form: keys used verbatim for explicit control.
    pre_warm : bool (optional)
        Trigger compilation of any `JitWrapper`-wrapped inner envs
        immediately. Default is `False` — call `multi_env.compile()` later.

    Returns
    -------
    multi_env : MultiEnv
        Manager holding all `JaxEnv` instances.
    """
    multi = MultiEnv(envs)

    if pre_warm:
        multi.compile()

    return multi


def make_multi_vec(
    envs: List[BatchedEnv] | Dict[str, BatchedEnv],
    *,
    jit_compile: bool = True,
    pre_warm: bool = False,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> MultiVecEnv:
    """
    Wrap `BatchedEnv` instances into a `MultiVecEnv`.

    Parameters
    ----------
    envs : List[BatchedEnv] | Dict[str, BatchedEnv]
        Envs to wrap. List form: keys derived from `env.name` with suffixes
        on duplicates. Dict form: keys used verbatim for explicit control.
    jit_compile : bool (optional)
        Enable the XLA compilation cache. Default is `True`.
    pre_warm : bool (optional)
        Compile each inner env and the multi-step jit immediately. Default
        is `False` — call `multi_vec_env.compile()` later.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.

    Returns
    -------
    multi_vec_env : MultiVecEnv
        Manager holding all `BatchedEnv` instances.
    """
    if jit_compile:
        setup_cache(cache_dir)

    multi = MultiVecEnv(envs)

    if jit_compile and pre_warm:
        multi.compile(cache_dir=cache_dir)

    return multi
