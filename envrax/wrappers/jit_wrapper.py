# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pathlib
from typing import Any, Dict, Tuple

import chex
import jax

from envrax._compile import DEFAULT_CACHE_DIR, setup_cache
from envrax.base import EnvParams, JaxEnv
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

    def reset(self, rng: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, Any]:
        return self._jit_reset(rng, params)

    def step(
        self,
        rng: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        return self._jit_step(rng, state, action, params)
