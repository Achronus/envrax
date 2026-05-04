import pathlib
from typing import Any, Dict, Tuple

import chex
import jax

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.env import ActSpaceT, ConfigT, JaxEnv, ObsSpaceT, StateT
from envrax.wrappers.base import Wrapper


class JitWrapper(Wrapper[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
    """
    Wrap a `JaxEnv` so that `reset` and `step` are compiled with
    `jax.jit` on construction.

    Parameters
    ----------
    env : JaxEnv
        Environment to wrap.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.
        Defaults to `~/.cache/envrax/xla_cache`. Pass `None` to disable.
    pre_warm : bool (optional)
        Run a dummy `reset` + `step` immediately to trigger XLA compilation.
        Set to `False` to defer compilation to the first real call or an
        explicit `compile()` call. Default is `True`.
    """

    def __init__(
        self,
        env: JaxEnv[ObsSpaceT, ActSpaceT, StateT, ConfigT],
        cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
        *,
        pre_warm: bool = True,
    ) -> None:
        super().__init__(env)
        setup_cache(cache_dir)

        self._jit_reset = jax.jit(env.reset)
        self._jit_step = jax.jit(env.step)

        if pre_warm:
            self.compile()

    def compile(self) -> None:
        """
        Trigger XLA compilation by running a dummy `reset` + `step`.

        Safe to call multiple times — subsequent calls are near-instant
        because JAX caches the compiled kernels in memory.
        """
        _key = jax.random.key(0)
        _, _state = self._jit_reset(_key)
        self._jit_step(_state, self._env.action_space.sample(_key))

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
        return self._jit_reset(rng)

    def step(
        self,
        state: StateT,
        action: chex.Array,
    ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
        return self._jit_step(state, action)
