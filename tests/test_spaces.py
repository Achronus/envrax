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

import jax
import jax.numpy as jnp
import pytest

from envrax.spaces import Box, Discrete


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
