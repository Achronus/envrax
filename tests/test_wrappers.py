import chex
import jax
import jax.numpy as jnp

from envrax.base import EnvConfig, EnvState, JaxEnv
from envrax.spaces import Box, Discrete, MultiDiscrete
from envrax.vec_env import VecEnv
from envrax.wrappers import (
    ClipReward,
    EpisodeDiscount,
    ExpandDims,
    FrameStackObservation,
    GrayscaleObservation,
    NormalizeObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
)

# ---------------------------------------------------------------------------
# Minimal pixel env for testing (returns uint8[4, 4, 3] RGB frames)
# ---------------------------------------------------------------------------


@chex.dataclass
class _PixelState(EnvState):
    pass


class _PixelEnv(JaxEnv[Box, Discrete, _PixelState]):
    """Minimal env returning uint8[4, 4, 3] RGB observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=255, shape=(4, 4, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng: chex.PRNGKey):
        obs = jnp.full((4, 4, 3), 128, dtype=jnp.uint8)
        state = _PixelState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, state, action):
        obs = jnp.full((4, 4, 3), 64, dtype=jnp.uint8)
        new_state = state.replace(
            step=state.step + 1,
            done=jnp.bool_(state.step + 1 >= self.config.max_steps),
        )
        reward = jnp.float32(1.0)
        done = new_state.done
        return obs, new_state, reward, done, {}


_RNG = jax.random.key(42)
_CONFIG = EnvConfig(max_steps=10)


def _env() -> _PixelEnv:
    return _PixelEnv(config=_CONFIG)


class TestClipReward:
    def test_positive_reward_clipped(self):
        env = ClipReward(_env())
        _, state = env.reset(_RNG)
        _, _, reward, _, _ = env.step(state, jnp.int32(0))
        assert float(reward) == 1.0  # sign(1.0) = 1.0

    def test_negative_reward_clipped(self):
        class _NegRewardEnv(_PixelEnv):
            def step(self, state, action):
                obs, new_state, _, done, info = super().step(state, action)
                return obs, new_state, jnp.float32(-5.0), done, info

        env = ClipReward(_NegRewardEnv(config=_CONFIG))
        _, state = env.reset(_RNG)
        _, _, reward, _, _ = env.step(state, jnp.int32(0))
        assert float(reward) == -1.0


class TestEpisodeDiscount:
    def test_discount_is_one_when_not_done(self):
        env = EpisodeDiscount(_env())
        _, state = env.reset(_RNG)
        _, _, _, discount, _ = env.step(state, jnp.int32(0))
        assert float(discount) == 1.0

    def test_discount_dtype(self):
        env = EpisodeDiscount(_env())
        _, state = env.reset(_RNG)
        _, _, _, discount, _ = env.step(state, jnp.int32(0))
        assert discount.dtype == jnp.float32


class TestExpandDims:
    def test_reward_shape(self):
        env = ExpandDims(_env())
        _, state = env.reset(_RNG)
        _, _, reward, done, _ = env.step(state, jnp.int32(0))
        assert reward.shape == (1,)
        assert done.shape == (1,)


class TestGrayscaleObservation:
    def test_obs_shape(self):
        env = GrayscaleObservation(_env())
        obs, _ = env.reset(_RNG)
        assert obs.shape == (4, 4)

    def test_obs_dtype(self):
        env = GrayscaleObservation(_env())
        obs, _ = env.reset(_RNG)
        assert obs.dtype == jnp.uint8

    def test_observation_space(self):
        env = GrayscaleObservation(_env())
        assert env.observation_space.shape == (4, 4)


class TestResizeObservation:
    def test_obs_shape(self):
        env = ResizeObservation(GrayscaleObservation(_env()), h=8, w=8)
        obs, _ = env.reset(_RNG)
        assert obs.shape == (8, 8)

    def test_observation_space(self):
        env = ResizeObservation(GrayscaleObservation(_env()), h=8, w=8)
        assert env.observation_space.shape == (8, 8)


class TestFrameStackObservation:
    def test_obs_shape(self):
        inner = ResizeObservation(GrayscaleObservation(_env()), h=4, w=4)
        env = FrameStackObservation(inner, n_stack=4)
        obs, _ = env.reset(_RNG)
        assert obs.shape == (4, 4, 4)

    def test_stack_updates_on_step(self):
        inner = ResizeObservation(GrayscaleObservation(_env()), h=4, w=4)
        env = FrameStackObservation(inner, n_stack=4)
        obs, state = env.reset(_RNG)
        obs2, _, _, _, _ = env.step(state, jnp.int32(0))
        assert obs2.shape == (4, 4, 4)


class TestNormalizeObservation:
    def test_obs_range(self):
        env = NormalizeObservation(_env())
        obs, _ = env.reset(_RNG)
        assert jnp.all(obs >= 0.0) and jnp.all(obs <= 1.0)

    def test_obs_dtype(self):
        env = NormalizeObservation(_env())
        obs, _ = env.reset(_RNG)
        assert obs.dtype == jnp.float32


class TestRecordEpisodeStatistics:
    def test_episode_key_in_info(self):
        env = RecordEpisodeStatistics(_env())
        _, state = env.reset(_RNG)
        _, _, _, _, info = env.step(state, jnp.int32(0))
        assert "episode" in info
        assert "r" in info["episode"]
        assert "l" in info["episode"]

    def test_return_accumulates(self):
        env = RecordEpisodeStatistics(_env())
        _, state = env.reset(_RNG)
        for _ in range(3):
            _, state, _, _, _ = env.step(state, jnp.int32(0))
        # episode_return should be accumulating (non-zero)
        assert float(state.episode_return) > 0.0


class TestVecEnv:
    def test_reset_batch_shape(self):
        env = VecEnv(_env(), num_envs=4)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (4, 4, 4, 3)

    def test_step_batch_shape(self):
        env = VecEnv(_env(), num_envs=4)
        _, states = env.reset(seed=42)
        actions = jnp.zeros(4, dtype=jnp.int32)
        obs, _, rewards, dones, _ = env.step(states, actions)
        assert obs.shape == (4, 4, 4, 3)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

    def test_single_observation_space(self):
        env = VecEnv(_env(), num_envs=8)
        space = env.single_observation_space
        assert isinstance(space, Box)
        assert space.shape == (4, 4, 3)

    def test_single_action_space(self):
        env = VecEnv(_env(), num_envs=8)
        space = env.single_action_space
        assert isinstance(space, Discrete)
        assert space.n == 2

    def test_observation_space_batched(self):
        env = VecEnv(_env(), num_envs=8)
        space = env.observation_space
        assert isinstance(space, Box)
        assert space.shape == (8, 4, 4, 3)

    def test_action_space_batched(self):
        env = VecEnv(_env(), num_envs=8)
        space = env.action_space
        assert isinstance(space, MultiDiscrete)
        assert space.nvec == (2,) * 8


class TestJitCompile:
    def test_jit_reset(self):
        env = _env()
        obs, _ = jax.jit(env.reset)(_RNG)
        assert obs.shape == (4, 4, 3)

    def test_jit_step(self):
        env = _env()
        _, state = env.reset(_RNG)
        obs, _, reward, _, _ = jax.jit(env.step)(state, jnp.int32(0))
        assert obs.shape == (4, 4, 3)
        assert reward.dtype == jnp.float32

    def test_jit_wrapped_env(self):
        env = GrayscaleObservation(_env())
        obs, state = jax.jit(env.reset)(_RNG)
        obs2, _, _, _, _ = jax.jit(env.step)(state, jnp.int32(0))
        assert obs.shape == (4, 4)
        assert obs2.shape == (4, 4)


class TestScanRollout:
    def test_scan_rollout(self):
        env = _env()
        _, state = env.reset(_RNG)

        def step_fn(state, _):
            obs, new_state, reward, done, _ = env.step(state, jnp.int32(0))
            return new_state, (obs, reward, done)

        _, (obs_seq, reward_seq, done_seq) = jax.lax.scan(
            step_fn, state, None, length=5
        )
        assert obs_seq.shape == (5, 4, 4, 3)
        assert reward_seq.shape == (5,)
        assert done_seq.shape == (5,)

    def test_scan_wrapped_env(self):
        env = GrayscaleObservation(_env())
        _, state = env.reset(_RNG)

        def step_fn(state, _):
            obs, new_state, _, _, _ = env.step(state, jnp.int32(0))
            return new_state, obs

        _, obs_seq = jax.lax.scan(step_fn, state, None, length=5)
        assert obs_seq.shape == (5, 4, 4)


class TestDeterminism:
    def test_reset_same_key_same_obs(self):
        env = _env()
        obs1, _ = env.reset(_RNG)
        obs2, _ = env.reset(_RNG)
        assert jnp.array_equal(obs1, obs2)

    def test_step_same_key_same_output(self):
        env = _env()
        _, state = env.reset(_RNG)
        obs1, _, r1, d1, _ = env.step(state, jnp.int32(0))
        obs2, _, r2, d2, _ = env.step(state, jnp.int32(0))
        assert jnp.array_equal(obs1, obs2)
        assert jnp.array_equal(r1, r2)
        assert jnp.array_equal(d1, d2)

    def test_different_keys_same_shape(self):
        env = _env()
        obs1, _ = env.reset(jax.random.key(0))
        obs2, _ = env.reset(jax.random.key(1))
        assert obs1.shape == obs2.shape


class TestVecEnvAutoReset:
    def test_auto_reset_when_done(self):
        env = VecEnv(_PixelEnv(config=EnvConfig(max_steps=1)), num_envs=1)
        _, states = env.reset(seed=0)
        actions = jnp.zeros(1, dtype=jnp.int32)
        _, new_states, _, done, _ = env.step(states, actions)
        assert bool(done[0])
        assert int(new_states.step[0]) == 0

    def test_no_reset_when_not_done(self):
        env = VecEnv(_PixelEnv(config=EnvConfig(max_steps=100)), num_envs=1)
        _, states = env.reset(seed=0)
        actions = jnp.zeros(1, dtype=jnp.int32)
        _, new_states, _, done, _ = env.step(states, actions)
        assert not bool(done[0])
        assert int(new_states.step[0]) == 1

    def test_auto_reset_jit_compatible(self):
        env = VecEnv(_env(), num_envs=2)
        _, states = env.reset(seed=0)
        actions = jnp.zeros(2, dtype=jnp.int32)
        obs, _, _, _, _ = jax.jit(env.step)(states, actions)
        assert obs.shape == (2, 4, 4, 3)
