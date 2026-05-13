import jax.numpy as jnp
import pytest

from envrax.spaces import Box, Discrete, MultiDiscrete
from envrax.utils import flat_size


class TestFlatSize:
    def test_box_1d(self):
        space = Box(low=0, high=1, shape=(7,), dtype=jnp.float32)
        assert flat_size(space, "env", "observation") == 7

    def test_box_multi_dim(self):
        space = Box(low=0, high=1, shape=(8, 8, 3), dtype=jnp.float32)
        assert flat_size(space, "env", "observation") == 192

    def test_box_scalar_shape(self):
        space = Box(low=0, high=1, shape=(), dtype=jnp.float32)
        assert flat_size(space, "env", "observation") == 1

    def test_returns_int(self):
        space = Box(low=0, high=1, shape=(4,), dtype=jnp.float32)
        assert isinstance(flat_size(space, "env", "action"), int)

    def test_discrete_raises(self):
        with pytest.raises(TypeError, match="has no `.shape`"):
            flat_size(Discrete(n=2), "env", "action")

    def test_multidiscrete_raises(self):
        with pytest.raises(TypeError, match="has no `.shape`"):
            flat_size(MultiDiscrete(nvec=(2, 3)), "env", "action")

    def test_error_message_includes_env_name_and_kind(self):
        with pytest.raises(TypeError, match="my/env-v0.*action.*Discrete"):
            flat_size(Discrete(n=2), "my/env-v0", "action")
