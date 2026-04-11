import pathlib
from typing import Any, Dict, Tuple

import chex
import jax

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.base import ActSpaceT, JaxEnv, ObsSpaceT, StateT
from envrax.wrappers.base import Wrapper


class JitWrapper(Wrapper[ObsSpaceT, ActSpaceT, StateT]):
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
    """

    def __init__(
        self,
        env: JaxEnv[ObsSpaceT, ActSpaceT, StateT],
        cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
    ) -> None:
        super().__init__(env)
        setup_cache(cache_dir)

        self._jit_reset = jax.jit(env.reset)
        self._jit_step = jax.jit(env.step)

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
        return self._jit_reset(rng)

    def step(
        self,
        state: StateT,
        action: chex.Array,
    ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
        return self._jit_step(state, action)
