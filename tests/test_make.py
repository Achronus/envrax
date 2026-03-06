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

"""Tests for the envrax make() factory functions.

Run with:
    pytest tests/test_make.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from envrax.base import EnvParams, EnvState, JaxEnv
from envrax.make import make, make_multi, make_multi_vec, make_vec
from envrax.registry import _REGISTRY, register
from envrax.spaces import Box, Discrete
from envrax.wrappers import GrayscaleObservation, JitWrapper, VmapEnv


# ---------------------------------------------------------------------------
# Minimal concrete env for testing
# ---------------------------------------------------------------------------


@chex.dataclass
class _PixState(EnvState):
    pass


class _PixEnv(JaxEnv):
    """Minimal env returning uint8[8, 8, 3] RGB observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=255, shape=(8, 8, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=4)

    def reset(self, rng: chex.PRNGKey, params: EnvParams):
        obs = jnp.full((8, 8, 3), 128, dtype=jnp.uint8)
        state = _PixState(step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, rng, state, action, params):
        obs = jnp.full((8, 8, 3), 64, dtype=jnp.uint8)
        new_state = state.replace(
            step=state.step + 1,
            done=jnp.bool_(state.step + 1 >= params.max_steps),
        )
        reward = jnp.float32(1.0)
        return obs, new_state, reward, new_state.done, {}


_RNG = jax.random.PRNGKey(0)
_PARAMS = EnvParams(max_steps=10)
_ENV_NAME = "TestPixEnv-v0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_and_cleanup():
    """Register the test env before each test and clean up after."""
    _REGISTRY.pop(_ENV_NAME, None)
    register(_ENV_NAME, _PixEnv, _PARAMS)
    yield
    _REGISTRY.pop(_ENV_NAME, None)


# ---------------------------------------------------------------------------
# make() tests
# ---------------------------------------------------------------------------


class TestMake:
    def test_returns_tuple(self):
        env, params = make(_ENV_NAME, jit_compile=False)
        assert isinstance(env, _PixEnv)
        assert isinstance(params, EnvParams)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            make("DoesNotExist-v0")

    def test_custom_params(self):
        custom = EnvParams(max_steps=500)
        _, params = make(_ENV_NAME, params=custom, jit_compile=False)
        assert params.max_steps == 500

    def test_default_params_from_registry(self):
        _, params = make(_ENV_NAME, jit_compile=False)
        assert params.max_steps == _PARAMS.max_steps

    def test_reset_shape(self):
        env, params = make(_ENV_NAME, jit_compile=False)
        obs, _ = env.reset(_RNG, params)
        chex.assert_shape(obs, (8, 8, 3))

    def test_step_shape(self):
        env, params = make(_ENV_NAME, jit_compile=False)
        _, state = env.reset(_RNG, params)
        obs, _, reward, done, _ = env.step(_RNG, state, jnp.int32(0), params)
        chex.assert_shape(obs, (8, 8, 3))
        chex.assert_rank(reward, 0)
        chex.assert_rank(done, 0)

    def test_wrappers_applied(self):
        env, _ = make(_ENV_NAME, wrappers=[GrayscaleObservation], jit_compile=False)
        assert isinstance(env, GrayscaleObservation)

    def test_wrapper_changes_obs_shape(self):
        env, params = make(_ENV_NAME, wrappers=[GrayscaleObservation], jit_compile=False)
        obs, _ = env.reset(_RNG, params)
        chex.assert_shape(obs, (8, 8))

    def test_jit_compile_returns_jit_wrapper(self):
        env, _ = make(_ENV_NAME, jit_compile=True)
        assert isinstance(env, JitWrapper)

    def test_jit_compile_reset_works(self):
        env, params = make(_ENV_NAME, jit_compile=True)
        obs, _ = env.reset(_RNG, params)
        chex.assert_shape(obs, (8, 8, 3))


# ---------------------------------------------------------------------------
# make_vec() tests
# ---------------------------------------------------------------------------


class TestMakeVec:
    def test_returns_vmap_env(self):
        vec_env, _ = make_vec(_ENV_NAME, n_envs=4, jit_compile=False)
        assert isinstance(vec_env, VmapEnv)
        assert vec_env.num_envs == 4

    def test_reset_shape(self):
        vec_env, params = make_vec(_ENV_NAME, n_envs=4, jit_compile=False)
        obs, _ = vec_env.reset(_RNG, params)
        chex.assert_shape(obs, (4, 8, 8, 3))

    def test_step_shape(self):
        vec_env, params = make_vec(_ENV_NAME, n_envs=4, jit_compile=False)
        _, states = vec_env.reset(_RNG, params)
        obs, _, rewards, dones, _ = vec_env.step(
            _RNG, states, jnp.zeros(4, dtype=jnp.int32), params
        )
        chex.assert_shape(obs, (4, 8, 8, 3))
        chex.assert_shape(rewards, (4,))
        chex.assert_shape(dones, (4,))

    def test_custom_params(self):
        custom = EnvParams(max_steps=200)
        _, params = make_vec(_ENV_NAME, n_envs=2, params=custom, jit_compile=False)
        assert params.max_steps == 200


# ---------------------------------------------------------------------------
# make_multi() tests
# ---------------------------------------------------------------------------


class TestMakeMulti:
    def test_returns_list_of_tuples(self):
        results = make_multi([_ENV_NAME, _ENV_NAME], jit_compile=False)
        assert len(results) == 2
        for env, params in results:
            assert isinstance(env, _PixEnv)
            assert isinstance(params, EnvParams)

    def test_empty_list_returns_empty(self):
        results = make_multi([], jit_compile=False)
        assert results == []


# ---------------------------------------------------------------------------
# make_multi_vec() tests
# ---------------------------------------------------------------------------


class TestMakeMultiVec:
    def test_returns_list_of_vmap_envs(self):
        results = make_multi_vec([_ENV_NAME, _ENV_NAME], n_envs=2, jit_compile=False)
        assert len(results) == 2
        for vec_env, params in results:
            assert isinstance(vec_env, VmapEnv)
            assert vec_env.num_envs == 2
