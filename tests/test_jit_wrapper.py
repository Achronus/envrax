"""Tests for JitWrapper.

Run with:
    pytest tests/test_jit_wrapper.py -v
"""

import chex
import jax
import jax.numpy as jnp

from envrax.env import EnvConfig, EnvState, JaxEnv
from envrax.spaces import Box, Discrete
from envrax.wrappers import JitWrapper, Wrapper

# ---------------------------------------------------------------------------
# Minimal concrete env for testing
# ---------------------------------------------------------------------------


@chex.dataclass
class _VectorState(EnvState):
    pass


class _VectorEnv(JaxEnv[Box, Discrete, _VectorState]):
    """Minimal env returning float32[4] observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=(4,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng: chex.PRNGKey):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        state = _VectorState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, state, action):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        new_state = state.__replace__(step=state.step + 1)
        reward = jnp.float32(1.0)
        done = new_state.step >= self.config.max_steps
        return obs, new_state.__replace__(done=done), reward, done, {}


_RNG = jax.random.key(0)
_CONFIG = EnvConfig(max_steps=10)


class TestJitWrapper:
    def test_jit_wrapper_is_wrapper(self):
        env = JitWrapper(_VectorEnv(config=_CONFIG), cache_dir=None)
        assert isinstance(env, Wrapper)

    def test_jit_wrapper_has_compiled_methods(self):
        env = JitWrapper(_VectorEnv(config=_CONFIG), cache_dir=None)
        assert hasattr(env, "_jit_reset")
        assert hasattr(env, "_jit_step")

    def test_reset_shape(self):
        env = JitWrapper(_VectorEnv(config=_CONFIG), cache_dir=None)
        obs, _ = env.reset(_RNG)
        chex.assert_shape(obs, (4,))
        assert obs.dtype == jnp.float32

    def test_step_shape(self):
        env = JitWrapper(_VectorEnv(config=_CONFIG), cache_dir=None)
        _, state = env.reset(_RNG)
        obs, _, reward, done, _ = env.step(state, jnp.int32(0))
        chex.assert_shape(obs, (4,))
        chex.assert_rank(reward, 0)
        chex.assert_rank(done, 0)

    def test_step_increments_step(self):
        env = JitWrapper(_VectorEnv(config=_CONFIG), cache_dir=None)
        _, state = env.reset(_RNG)
        _, state2, _, _, _ = env.step(state, jnp.int32(0))
        assert int(state2.step) == 1

    def test_spaces_delegate_to_inner(self):
        inner = _VectorEnv(config=_CONFIG)
        env = JitWrapper(inner, cache_dir=None)
        assert env.observation_space == inner.observation_space
        assert env.action_space == inner.action_space

    def test_unwrapped_returns_base_env(self):
        inner = _VectorEnv(config=_CONFIG)
        env = JitWrapper(inner, cache_dir=None)
        assert env.unwrapped is inner

    def test_repr_contains_class_name(self):
        env = JitWrapper(_VectorEnv(config=_CONFIG), cache_dir=None)
        assert "JitWrapper" in repr(env)

    def test_done_after_max_steps(self):
        env = JitWrapper(_VectorEnv(config=EnvConfig(max_steps=2)), cache_dir=None)
        _, state = env.reset(_RNG)
        _, state, _, _, _ = env.step(state, jnp.int32(0))
        _, _, _, done, _ = env.step(state, jnp.int32(0))
        assert bool(done)
