import jax
import jax.numpy as jnp

from envrax.env import EnvConfig, EnvState

_RNG = jax.random.key(0)


class TestEnvState:
    def test_fields(self):
        state = EnvState(rng=_RNG, step=jnp.int32(0), done=jnp.bool_(False))
        assert int(state.step) == 0
        assert not bool(state.done)

    def test_replace(self):
        state = EnvState(rng=_RNG, step=jnp.int32(0), done=jnp.bool_(False))
        new_state = state.__replace__(step=jnp.int32(5))
        assert int(new_state.step) == 5
        assert not bool(new_state.done)


class TestEnvConfig:
    def test_default(self):
        config = EnvConfig()
        assert config.max_steps == 1000

    def test_custom(self):
        config = EnvConfig(max_steps=500)
        assert config.max_steps == 500

    def test_replace(self):
        config = EnvConfig()
        new_config = config.__replace__(max_steps=2000)
        assert new_config.max_steps == 2000
