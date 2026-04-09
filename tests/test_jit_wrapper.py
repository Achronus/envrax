"""Tests for JitWrapper.

Run with:
    pytest tests/test_jit_wrapper.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from envrax.base import EnvParams, EnvState, JaxEnv
from envrax.spaces import Box, Discrete
from envrax.wrappers import JitWrapper, Wrapper


# ---------------------------------------------------------------------------
# Minimal concrete env for testing
# ---------------------------------------------------------------------------


@chex.dataclass
class _VectorState(EnvState):
    pass


class _VectorEnv(JaxEnv):
    """Minimal env returning float32[4] observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=(4,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng: chex.PRNGKey, params: EnvParams):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        state = _VectorState(step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, rng, state, action, params):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        new_state = state.replace(step=state.step + 1)
        reward = jnp.float32(1.0)
        done = new_state.step >= params.max_steps
        return obs, new_state.replace(done=done), reward, done, {}


_RNG = jax.random.PRNGKey(0)
_PARAMS = EnvParams(max_steps=10)


class TestJitWrapper:
    def test_jit_wrapper_is_wrapper(self):
        env = JitWrapper(_VectorEnv(), cache_dir=None)
        assert isinstance(env, Wrapper)

    def test_jit_wrapper_has_compiled_methods(self):
        env = JitWrapper(_VectorEnv(), cache_dir=None)
        assert hasattr(env, "_jit_reset")
        assert hasattr(env, "_jit_step")

    def test_reset_shape(self):
        env = JitWrapper(_VectorEnv(), cache_dir=None)
        obs, _ = env.reset(_RNG, _PARAMS)
        chex.assert_shape(obs, (4,))
        assert obs.dtype == jnp.float32

    def test_step_shape(self):
        env = JitWrapper(_VectorEnv(), cache_dir=None)
        _, state = env.reset(_RNG, _PARAMS)
        obs, _, reward, done, _ = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        chex.assert_shape(obs, (4,))
        chex.assert_rank(reward, 0)
        chex.assert_rank(done, 0)

    def test_step_increments_step(self):
        env = JitWrapper(_VectorEnv(), cache_dir=None)
        _, state = env.reset(_RNG, _PARAMS)
        _, state2, _, _, _ = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        assert int(state2.step) == 1

    def test_spaces_delegate_to_inner(self):
        inner = _VectorEnv()
        env = JitWrapper(inner, cache_dir=None)
        assert env.observation_space == inner.observation_space
        assert env.action_space == inner.action_space

    def test_unwrapped_returns_base_env(self):
        inner = _VectorEnv()
        env = JitWrapper(inner, cache_dir=None)
        assert env.unwrapped is inner

    def test_repr_contains_class_name(self):
        env = JitWrapper(_VectorEnv(), cache_dir=None)
        assert "JitWrapper" in repr(env)

    def test_done_after_max_steps(self):
        env = JitWrapper(_VectorEnv(), cache_dir=None)
        params = EnvParams(max_steps=2)
        _, state = env.reset(_RNG, params)
        _, state, _, _, _ = env.step(_RNG, state, jnp.int32(0), params)
        _, _, _, done, _ = env.step(_RNG, state, jnp.int32(0), params)
        assert bool(done)
