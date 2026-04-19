import jax
import jax.numpy as jnp
import pytest

from envrax.spaces import Box, Discrete, MultiDiscrete, batch_space


class TestDiscrete:
    def test_sample_shape(self):
        space = Discrete(n=6)
        rng = jax.random.PRNGKey(0)
        action = space.sample(rng)
        assert action.shape == ()

    def test_sample_dtype(self):
        space = Discrete(n=6)
        rng = jax.random.PRNGKey(0)
        action = space.sample(rng)
        assert action.dtype == jnp.int32

    def test_sample_range(self):
        space = Discrete(n=6)
        for i in range(20):
            rng = jax.random.PRNGKey(i)
            action = space.sample(rng)
            assert 0 <= int(action) < 6

    def test_contains_valid(self):
        space = Discrete(n=6)
        assert space.contains(jnp.int32(0))
        assert space.contains(jnp.int32(5))

    def test_contains_invalid(self):
        space = Discrete(n=6)
        assert not space.contains(jnp.int32(6))
        assert not space.contains(jnp.int32(-1))

    def test_default_dtype(self):
        space = Discrete(n=6)
        assert space.dtype == jnp.int32

    def test_custom_dtype(self):
        space = Discrete(n=6, dtype=jnp.int16)
        rng = jax.random.PRNGKey(0)
        action = space.sample(rng)
        assert action.dtype == jnp.int16


class TestBox:
    def test_sample_shape(self):
        space = Box(low=0, high=255, shape=(84, 84, 1), dtype=jnp.uint8)
        rng = jax.random.PRNGKey(0)
        obs = space.sample(rng)
        assert obs.shape == (84, 84, 1)

    def test_sample_dtype(self):
        space = Box(low=0, high=255, shape=(84, 84, 1), dtype=jnp.uint8)
        rng = jax.random.PRNGKey(0)
        obs = space.sample(rng)
        assert obs.dtype == jnp.uint8

    def test_sample_range_integer(self):
        space = Box(low=0, high=255, shape=(10,), dtype=jnp.uint8)
        rng = jax.random.PRNGKey(0)
        obs = space.sample(rng)
        assert jnp.all(obs >= 0) and jnp.all(obs <= 255)

    def test_sample_range_float(self):
        space = Box(low=0.0, high=1.0, shape=(10,), dtype=jnp.float32)
        rng = jax.random.PRNGKey(0)
        obs = space.sample(rng)
        assert jnp.all(obs >= 0.0) and jnp.all(obs <= 1.0)

    def test_contains_valid(self):
        space = Box(low=0, high=255, shape=(4,), dtype=jnp.uint8)
        x = jnp.array([0, 128, 255, 100], dtype=jnp.uint8)
        assert space.contains(x)

    def test_contains_wrong_shape(self):
        space = Box(low=0, high=255, shape=(4,), dtype=jnp.uint8)
        x = jnp.array([0, 128], dtype=jnp.uint8)
        assert not space.contains(x)


class TestMultiDiscrete:
    def test_sample_shape(self):
        space = MultiDiscrete(nvec=(4, 6, 3))
        rng = jax.random.PRNGKey(0)
        actions = space.sample(rng)
        assert actions.shape == (3,)

    def test_sample_dtype(self):
        space = MultiDiscrete(nvec=(4, 6, 3))
        rng = jax.random.PRNGKey(0)
        actions = space.sample(rng)
        assert actions.dtype == jnp.int32

    def test_sample_range(self):
        space = MultiDiscrete(nvec=(4, 6, 3))
        for i in range(20):
            rng = jax.random.PRNGKey(i)
            actions = space.sample(rng)
            assert int(actions[0]) < 4
            assert int(actions[1]) < 6
            assert int(actions[2]) < 3
            assert jnp.all(actions >= 0)

    def test_contains_valid(self):
        space = MultiDiscrete(nvec=(4, 6))
        assert space.contains(jnp.array([0, 5], dtype=jnp.int32))
        assert space.contains(jnp.array([3, 0], dtype=jnp.int32))

    def test_contains_invalid(self):
        space = MultiDiscrete(nvec=(4, 6))
        assert not space.contains(jnp.array([4, 0], dtype=jnp.int32))
        assert not space.contains(jnp.array([-1, 0], dtype=jnp.int32))

    def test_contains_wrong_shape(self):
        space = MultiDiscrete(nvec=(4, 6))
        assert not space.contains(jnp.array([1], dtype=jnp.int32))

    def test_uniform_nvec(self):
        """MultiDiscrete with identical sub-spaces (batched Discrete)."""
        space = MultiDiscrete(nvec=(5, 5, 5, 5))
        rng = jax.random.PRNGKey(42)
        actions = space.sample(rng)
        assert actions.shape == (4,)
        assert jnp.all(actions < 5)

    def test_default_dtype(self):
        space = MultiDiscrete(nvec=(4, 6, 3))
        assert space.dtype == jnp.int32

    def test_custom_dtype(self):
        space = MultiDiscrete(nvec=(4, 6, 3), dtype=jnp.int16)
        rng = jax.random.PRNGKey(0)
        actions = space.sample(rng)
        assert actions.dtype == jnp.int16


class TestBatchSpace:
    def test_discrete_to_multi_discrete(self):
        batched = batch_space(Discrete(n=4), 8)
        assert isinstance(batched, MultiDiscrete)
        assert batched.nvec == (4,) * 8

    def test_box_prepends_dimension(self):
        batched = batch_space(Box(low=0.0, high=1.0, shape=(84, 84, 3), dtype=jnp.float32), 16)
        assert isinstance(batched, Box)
        assert batched.shape == (16, 84, 84, 3)
        assert batched.low == 0.0
        assert batched.high == 1.0

    def test_multi_discrete_repeats(self):
        batched = batch_space(MultiDiscrete(nvec=(3, 5)), 4)
        assert isinstance(batched, MultiDiscrete)
        assert batched.nvec == (3, 5) * 4

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            batch_space("not a space", 4)

    def test_discrete_preserves_dtype(self):
        batched = batch_space(Discrete(n=4, dtype=jnp.int64), 8)
        assert isinstance(batched, MultiDiscrete)
        assert batched.dtype == jnp.int64

    def test_multi_discrete_preserves_dtype(self):
        batched = batch_space(MultiDiscrete(nvec=(3, 5), dtype=jnp.int64), 4)
        assert isinstance(batched, MultiDiscrete)
        assert batched.dtype == jnp.int64

    def test_box_preserves_dtype(self):
        batched = batch_space(
            Box(low=0.0, high=1.0, shape=(4,), dtype=jnp.float64), 8
        )
        assert isinstance(batched, Box)
        assert batched.dtype == jnp.float64
