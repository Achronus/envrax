import jax
import jax.numpy as jnp
import pytest

from envrax.base import EnvParams, EnvState


class TestEnvState:
    def test_fields(self):
        state = EnvState(step=jnp.int32(0), done=jnp.bool_(False))
        assert int(state.step) == 0
        assert not bool(state.done)

    def test_replace(self):
        state = EnvState(step=jnp.int32(0), done=jnp.bool_(False))
        new_state = state.replace(step=jnp.int32(5))
        assert int(new_state.step) == 5
        assert not bool(new_state.done)


class TestEnvParams:
    def test_default(self):
        params = EnvParams()
        assert params.max_steps == 1000

    def test_custom(self):
        params = EnvParams(max_steps=500)
        assert params.max_steps == 500

    def test_replace(self):
        params = EnvParams()
        new_params = params.replace(max_steps=2000)
        assert new_params.max_steps == 2000
