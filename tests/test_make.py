"""Tests for the envrax make() factory functions.

Run with:
    pytest tests/test_make.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from envrax.base import EnvConfig, EnvState, JaxEnv
from envrax.make import make, make_multi, make_multi_vec, make_vec
from envrax.registry import _REGISTRY, register
from envrax.spaces import Box, Discrete
from envrax.vec_env import VecEnv
from envrax.wrappers import GrayscaleObservation, JitWrapper

# ---------------------------------------------------------------------------
# Minimal concrete env for testing
# ---------------------------------------------------------------------------


@chex.dataclass
class _PixState(EnvState):
    pass


class _PixEnv(JaxEnv[Box, Discrete, _PixState]):
    """Minimal env returning uint8[8, 8, 3] RGB observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=255, shape=(8, 8, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=4)

    def reset(self, rng: chex.PRNGKey):
        obs = jnp.full((8, 8, 3), 128, dtype=jnp.uint8)
        state = _PixState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, state, action):
        obs = jnp.full((8, 8, 3), 64, dtype=jnp.uint8)
        new_state = state.replace(
            step=state.step + 1,
            done=jnp.bool_(state.step + 1 >= self.config.max_steps),
        )
        reward = jnp.float32(1.0)
        return obs, new_state, reward, new_state.done, {}


_RNG = jax.random.key(0)
_CONFIG = EnvConfig(max_steps=10)
_ENV_NAME = "TestPixEnv-v0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_and_cleanup():
    """Register the test env before each test and clean up after."""
    _REGISTRY.pop(_ENV_NAME, None)
    register(_ENV_NAME, _PixEnv, _CONFIG)
    yield
    _REGISTRY.pop(_ENV_NAME, None)


# ---------------------------------------------------------------------------
# make() tests
# ---------------------------------------------------------------------------


class TestMake:
    def test_returns_tuple(self):
        env, config = make(_ENV_NAME, jit_compile=False)
        assert isinstance(env, _PixEnv)
        assert isinstance(config, EnvConfig)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            make("DoesNotExist-v0")

    def test_custom_config(self):
        custom = EnvConfig(max_steps=500)
        _, config = make(_ENV_NAME, config=custom, jit_compile=False)
        assert config.max_steps == 500

    def test_default_config_from_registry(self):
        _, config = make(_ENV_NAME, jit_compile=False)
        assert config.max_steps == _CONFIG.max_steps

    def test_reset_shape(self):
        env, _ = make(_ENV_NAME, jit_compile=False)
        obs, _ = env.reset(_RNG)
        chex.assert_shape(obs, (8, 8, 3))

    def test_step_shape(self):
        env, _ = make(_ENV_NAME, jit_compile=False)
        _, state = env.reset(_RNG)
        obs, _, reward, done, _ = env.step(state, jnp.int32(0))
        chex.assert_shape(obs, (8, 8, 3))
        chex.assert_rank(reward, 0)
        chex.assert_rank(done, 0)

    def test_wrappers_applied(self):
        env, _ = make(_ENV_NAME, wrappers=[GrayscaleObservation], jit_compile=False)
        assert isinstance(env, GrayscaleObservation)

    def test_wrapper_changes_obs_shape(self):
        env, _ = make(
            _ENV_NAME, wrappers=[GrayscaleObservation], jit_compile=False
        )
        obs, _ = env.reset(_RNG)
        chex.assert_shape(obs, (8, 8))

    def test_jit_compile_returns_jit_wrapper(self):
        env, _ = make(_ENV_NAME, jit_compile=True)
        assert isinstance(env, JitWrapper)

    def test_jit_compile_reset_works(self):
        env, _ = make(_ENV_NAME, jit_compile=True)
        obs, _ = env.reset(_RNG)
        chex.assert_shape(obs, (8, 8, 3))


# ---------------------------------------------------------------------------
# make_vec() tests
# ---------------------------------------------------------------------------


class TestMakeVec:
    def test_returns_vec_env(self):
        vec_env, _ = make_vec(_ENV_NAME, n_envs=4, jit_compile=False)
        assert isinstance(vec_env, VecEnv)
        assert vec_env.num_envs == 4

    def test_reset_shape(self):
        vec_env, _ = make_vec(_ENV_NAME, n_envs=4, jit_compile=False)
        obs, _ = vec_env.reset(jax.random.key(0))
        chex.assert_shape(obs, (4, 8, 8, 3))

    def test_step_shape(self):
        vec_env, _ = make_vec(_ENV_NAME, n_envs=4, jit_compile=False)
        _, states = vec_env.reset(jax.random.key(0))
        obs, _, rewards, dones, _ = vec_env.step(
            states, jnp.zeros(4, dtype=jnp.int32)
        )
        chex.assert_shape(obs, (4, 8, 8, 3))
        chex.assert_shape(rewards, (4,))
        chex.assert_shape(dones, (4,))

    def test_custom_config(self):
        custom = EnvConfig(max_steps=200)
        _, config = make_vec(_ENV_NAME, n_envs=2, config=custom, jit_compile=False)
        assert config.max_steps == 200


# ---------------------------------------------------------------------------
# make_multi() tests
# ---------------------------------------------------------------------------


class TestMakeMulti:
    def test_returns_list_of_tuples(self):
        results = make_multi([_ENV_NAME, _ENV_NAME], jit_compile=False)
        assert len(results) == 2
        for env, config in results:
            assert isinstance(env, _PixEnv)
            assert isinstance(config, EnvConfig)

    def test_empty_list_returns_empty(self):
        results = make_multi([], jit_compile=False)
        assert results == []


# ---------------------------------------------------------------------------
# make_multi_vec() tests
# ---------------------------------------------------------------------------


class TestMakeMultiVec:
    def test_returns_list_of_vec_envs(self):
        results = make_multi_vec([_ENV_NAME, _ENV_NAME], n_envs=2, jit_compile=False)
        assert len(results) == 2
        for vec_env, _ in results:
            assert isinstance(vec_env, VecEnv)
            assert vec_env.num_envs == 2
