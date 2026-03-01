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

from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.base import EnvParams, JaxEnv
from envrax.spaces import Box
from envrax.wrappers.base import Wrapper


class NormalizeObservation(Wrapper):
    """
    Normalises pixel observations from `uint8 [0, 255]` to `float32 [0, 1]`.

    Divides observations by 255.0 and casts to float32.

    Parameters
    ----------
    env : JaxEnv
        Environment to wrap.
    """

    def __init__(self, env: JaxEnv) -> None:
        super().__init__(env)

    def reset(self, rng: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, Any]:
        """
        Reset and return a normalised initial observation.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        params : EnvParams
            Environment parameters.

        Returns
        -------
        obs  : chex.Array
            Normalised observation, float32 in [0, 1].
        state : Any
            Inner environment state.
        """
        obs, state = self._env.reset(rng, params)
        return obs.astype(jnp.float32) / jnp.float32(255.0), state

    def step(
        self,
        rng: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step and return a normalised observation.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : Any
            Current environment state.
        action : chex.Array
            Action to take.
        params : EnvParams
            Environment parameters.

        Returns
        -------
        obs  : chex.Array
            Normalised observation, float32 in [0, 1].
        new_state : Any
            Updated environment state.
        reward  : chex.Array
            Step reward.
        done  : chex.Array
            Terminal flag.
        info : Dict[str, Any]
            Environment metadata.
        """
        obs, new_state, reward, done, info = self._env.step(rng, state, action, params)
        return (
            obs.astype(jnp.float32) / jnp.float32(255.0),
            new_state,
            reward,
            done,
            info,
        )

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space
        return Box(low=0.0, high=1.0, shape=inner.shape, dtype=jnp.float32)
