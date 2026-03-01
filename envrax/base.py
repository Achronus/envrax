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

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import chex
import jax

from envrax.spaces import Space


@chex.dataclass
class EnvState:
    """
    Base environment state. Every environment extends this with its own fields.

    All fields must be JAX arrays or Python scalars (for static shape info).
    No Python objects, no lists, no dicts.

    Parameters
    ----------
    step : chex.Array
        Current timestep within the episode.
    done : chex.Array
        bool scalar — episode termination flag.
    """

    step: chex.Array
    done: chex.Array


@chex.dataclass
class EnvParams:
    """
    Static environment configuration. Set once at construction, never changed.
    Controls things like max_steps, reward scaling, difficulty, etc.

    Parameters
    ----------
    max_steps : int
        Maximum number of steps per episode. Default is 1000.
    """

    max_steps: int = 1000


class JaxEnv(ABC):
    """
    Base class for all JAX-native environments.

    Every method is a pure function — no side effects, no mutation.
    State is passed explicitly; environments are stateless Python objects.
    """

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """Returns the observation space (Box or Discrete)."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Returns the action space (always Discrete for these suites)."""
        ...

    @abstractmethod
    def reset(
        self,
        rng: chex.PRNGKey,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState]:
        """
        Pure function. Returns (observation, initial_state).

        All randomness flows through rng — no global state.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        params : EnvParams
            Static environment configuration.

        Returns
        -------
        obs : chex.Array
            Initial observation.
        state : EnvState
            Initial environment state.
        """
        ...

    @abstractmethod
    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Pure function. Returns (obs, new_state, reward, done, info).

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : EnvState
            Current environment state.
        action : chex.Array
            Integer scalar in [0, num_actions).
        params : EnvParams
            Static environment configuration.

        Returns
        -------
        obs : chex.Array
            Observation after the step.
        new_state : EnvState
            Updated environment state.
        reward : chex.Array
            float32 scalar reward.
        done : chex.Array
            bool scalar — True when the episode has ended.
        info : dict
            Auxiliary diagnostic information.
        """
        ...

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Wraps step() to auto-reset on episode end.

        When done is True, returns the observation from a fresh reset
        instead of the terminal observation. Override only if custom
        reset behaviour is needed.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : EnvState
            Current environment state.
        action : chex.Array
            Action to take.
        params : EnvParams
            Static environment configuration.

        Returns
        -------
        Tuple of (obs, new_state, reward, done, info), where obs/new_state
        come from reset if done is True.
        """
        obs, new_state, reward, done, info = self.step(rng, state, action, params)
        reset_rng, _ = jax.random.split(rng)
        reset_obs, reset_state = self.reset(reset_rng, params)
        final_obs = jax.lax.cond(done, lambda: reset_obs, lambda: obs)
        final_state = jax.lax.cond(done, lambda: reset_state, lambda: new_state)
        return final_obs, final_state, reward, done, info

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
