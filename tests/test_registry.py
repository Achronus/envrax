import chex
import jax
import jax.numpy as jnp
import pytest

from envrax.base import EnvConfig, EnvState, JaxEnv
from envrax.registry import _REGISTRY, make_env, register
from envrax.spaces import Box, Discrete

# ---------------------------------------------------------------------------
# Minimal concrete env for testing
# ---------------------------------------------------------------------------


@chex.dataclass
class _DummyState(EnvState):
    pass


class _DummyEnv(JaxEnv):
    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=(4,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng: chex.PRNGKey, config: EnvConfig):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        state = _DummyState(step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, rng, state, action, config):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        new_state = state.replace(step=state.step + 1)
        reward = jnp.float32(0.0)
        done = new_state.step >= config.max_steps
        return obs, new_state, reward, done, {}


class TestRegistry:
    def setup_method(self):
        # Clean slate for each test by removing our test env if registered
        _REGISTRY.pop("DummyEnv-v0", None)

    def teardown_method(self):
        _REGISTRY.pop("DummyEnv-v0", None)

    def test_register_and_make_env(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        env, config = make_env("DummyEnv-v0")
        assert isinstance(env, _DummyEnv)
        assert config.max_steps == 1000

    def test_make_with_override(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        env, config = make_env("DummyEnv-v0", max_steps=500)
        assert config.max_steps == 500

    def test_make_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            make_env("DoesNotExist-v0")

    def test_register_duplicate_raises(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        with pytest.raises(ValueError, match="already registered"):
            register("DummyEnv-v0", _DummyEnv, EnvConfig())

    def test_env_reset_step(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        env, config = make_env("DummyEnv-v0")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng, config)
        assert obs.shape == (4,)
        obs2, state2, reward, done, info = env.step(rng, state, jnp.int32(0), config)
        assert obs2.shape == (4,)
        assert int(state2.step) == 1
