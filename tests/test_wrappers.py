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

import chex
import jax
import jax.numpy as jnp
import pytest

from envrax.base import EnvParams, EnvState, JaxEnv
from envrax.spaces import Box, Discrete
from envrax.wrappers import (
    ClipReward,
    EpisodeDiscount,
    ExpandDims,
    FrameStackObservation,
    GrayscaleObservation,
    NormalizeObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
    VmapEnv,
)

# ---------------------------------------------------------------------------
# Minimal pixel env for testing (returns uint8[4, 4, 3] RGB frames)
# ---------------------------------------------------------------------------


@chex.dataclass
class _PixelState(EnvState):
    pass


class _PixelEnv(JaxEnv):
    """Minimal env returning uint8[4, 4, 3] RGB observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=255, shape=(4, 4, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng: chex.PRNGKey, params: EnvParams):
        obs = jnp.full((4, 4, 3), 128, dtype=jnp.uint8)
        state = _PixelState(step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, rng, state, action, params):
        obs = jnp.full((4, 4, 3), 64, dtype=jnp.uint8)
        new_state = state.replace(
            step=state.step + 1,
            done=jnp.bool_(state.step + 1 >= params.max_steps),
        )
        reward = jnp.float32(1.0)
        done = new_state.done
        return obs, new_state, reward, done, {}


_RNG = jax.random.PRNGKey(42)
_PARAMS = EnvParams(max_steps=10)


class TestClipReward:
    def test_positive_reward_clipped(self):
        env = ClipReward(_PixelEnv())
        _, state = env.reset(_RNG, _PARAMS)
        _, _, reward, _, _ = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        assert float(reward) == 1.0  # sign(1.0) = 1.0

    def test_negative_reward_clipped(self):
        class _NegRewardEnv(_PixelEnv):
            def step(self, rng, state, action, params):
                obs, new_state, _, done, info = super().step(rng, state, action, params)
                return obs, new_state, jnp.float32(-5.0), done, info

        env = ClipReward(_NegRewardEnv())
        _, state = env.reset(_RNG, _PARAMS)
        _, _, reward, _, _ = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        assert float(reward) == -1.0


class TestEpisodeDiscount:
    def test_discount_is_one_when_not_done(self):
        env = EpisodeDiscount(_PixelEnv())
        _, state = env.reset(_RNG, _PARAMS)
        _, _, _, discount, _ = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        assert float(discount) == 1.0

    def test_discount_dtype(self):
        env = EpisodeDiscount(_PixelEnv())
        _, state = env.reset(_RNG, _PARAMS)
        _, _, _, discount, _ = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        assert discount.dtype == jnp.float32


class TestExpandDims:
    def test_reward_shape(self):
        env = ExpandDims(_PixelEnv())
        _, state = env.reset(_RNG, _PARAMS)
        _, _, reward, done, _ = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        assert reward.shape == (1,)
        assert done.shape == (1,)


class TestGrayscaleObservation:
    def test_obs_shape(self):
        env = GrayscaleObservation(_PixelEnv())
        obs, _ = env.reset(_RNG, _PARAMS)
        assert obs.shape == (4, 4)

    def test_obs_dtype(self):
        env = GrayscaleObservation(_PixelEnv())
        obs, _ = env.reset(_RNG, _PARAMS)
        assert obs.dtype == jnp.uint8

    def test_observation_space(self):
        env = GrayscaleObservation(_PixelEnv())
        assert env.observation_space.shape == (4, 4)


class TestResizeObservation:
    def test_obs_shape(self):
        env = ResizeObservation(GrayscaleObservation(_PixelEnv()), h=8, w=8)
        obs, _ = env.reset(_RNG, _PARAMS)
        assert obs.shape == (8, 8)

    def test_observation_space(self):
        env = ResizeObservation(GrayscaleObservation(_PixelEnv()), h=8, w=8)
        assert env.observation_space.shape == (8, 8)


class TestFrameStackObservation:
    def test_obs_shape(self):
        inner = ResizeObservation(GrayscaleObservation(_PixelEnv()), h=4, w=4)
        env = FrameStackObservation(inner, n_stack=4)
        obs, _ = env.reset(_RNG, _PARAMS)
        assert obs.shape == (4, 4, 4)

    def test_stack_updates_on_step(self):
        inner = ResizeObservation(GrayscaleObservation(_PixelEnv()), h=4, w=4)
        env = FrameStackObservation(inner, n_stack=4)
        obs, state = env.reset(_RNG, _PARAMS)
        obs2, _, _, _, _ = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        assert obs2.shape == (4, 4, 4)


class TestNormalizeObservation:
    def test_obs_range(self):
        env = NormalizeObservation(_PixelEnv())
        obs, _ = env.reset(_RNG, _PARAMS)
        assert jnp.all(obs >= 0.0) and jnp.all(obs <= 1.0)

    def test_obs_dtype(self):
        env = NormalizeObservation(_PixelEnv())
        obs, _ = env.reset(_RNG, _PARAMS)
        assert obs.dtype == jnp.float32


class TestRecordEpisodeStatistics:
    def test_episode_key_in_info(self):
        env = RecordEpisodeStatistics(_PixelEnv())
        _, state = env.reset(_RNG, _PARAMS)
        _, _, _, _, info = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        assert "episode" in info
        assert "r" in info["episode"]
        assert "l" in info["episode"]

    def test_return_accumulates(self):
        env = RecordEpisodeStatistics(_PixelEnv())
        _, state = env.reset(_RNG, _PARAMS)
        for _ in range(3):
            _, state, _, _, _ = env.step(_RNG, state, jnp.int32(0), _PARAMS)
        # episode_return should be accumulating (non-zero)
        assert float(state.episode_return) > 0.0


class TestVmapEnv:
    def test_reset_batch_shape(self):
        env = VmapEnv(_PixelEnv(), num_envs=4)
        obs, states = env.reset(_RNG, _PARAMS)
        assert obs.shape == (4, 4, 4, 3)

    def test_step_batch_shape(self):
        env = VmapEnv(_PixelEnv(), num_envs=4)
        _, states = env.reset(_RNG, _PARAMS)
        actions = jnp.zeros(4, dtype=jnp.int32)
        obs, _, rewards, dones, _ = env.step(_RNG, states, actions, _PARAMS)
        assert obs.shape == (4, 4, 4, 3)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
