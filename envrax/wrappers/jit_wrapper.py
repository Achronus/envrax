import pathlib
from typing import Any, Dict, Tuple

import chex
import jax

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.base import EnvConfig, JaxEnv
from envrax.wrappers.base import Wrapper


class JitWrapper(Wrapper):
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
        env: JaxEnv,
        cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
    ) -> None:
        super().__init__(env)
        setup_cache(cache_dir)

        self._jit_reset = jax.jit(env.reset)
        self._jit_step = jax.jit(env.step)

    def reset(self, rng: chex.PRNGKey, config: EnvConfig) -> Tuple[chex.Array, Any]:
        return self._jit_reset(rng, config)

    def step(
        self,
        rng: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        config: EnvConfig,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        return self._jit_step(rng, state, action, config)
