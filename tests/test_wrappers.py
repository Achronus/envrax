import chex
import jax
import jax.numpy as jnp

from envrax.env import EnvConfig, EnvState, JaxEnv
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


class _PixelEnv(JaxEnv[Box, Discrete, _PixelState, EnvConfig]):
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
        new_state = state.__replace__(
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

    def test_obs_shape_preserves_rgb_channel(self):
        # Skip grayscale — resize directly on uint8[H, W, 3]
        env = ResizeObservation(_env(), h=8, w=8)
        obs, _ = env.reset(_RNG)
        assert obs.shape == (8, 8, 3)

    def test_observation_space_preserves_rgb_channel(self):
        env = ResizeObservation(_env(), h=8, w=8)
        assert env.observation_space.shape == (8, 8, 3)


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

    def test_observation_space_appends_stack_dim(self):
        inner = ResizeObservation(GrayscaleObservation(_env()), h=4, w=4)
        env = FrameStackObservation(inner, n_stack=4)
        space = env.observation_space
        assert isinstance(space, Box)
        assert space.shape == (4, 4, 4)


class TestNormalizeObservation:
    def test_obs_range(self):
        env = NormalizeObservation(_env())
        obs, _ = env.reset(_RNG)
        assert jnp.all(obs >= 0.0) and jnp.all(obs <= 1.0)

    def test_obs_dtype(self):
        env = NormalizeObservation(_env())
        obs, _ = env.reset(_RNG)
        assert obs.dtype == jnp.float32

    def test_step_normalises_observation(self):
        env = NormalizeObservation(_env())
        _, state = env.reset(_RNG)
        obs, _, reward, done, _ = env.step(state, jnp.int32(0))
        assert obs.dtype == jnp.float32
        assert jnp.all(obs >= 0.0) and jnp.all(obs <= 1.0)
        # _PixelEnv.step returns 64 → 64/255 ≈ 0.251
        assert jnp.allclose(obs, 64.0 / 255.0)

    def test_observation_space_is_float_unit_box(self):
        env = NormalizeObservation(_env())
        space = env.observation_space
        assert isinstance(space, Box)
        assert space.shape == (4, 4, 3)
        assert space.dtype == jnp.float32
        assert space.low == 0.0
        assert space.high == 1.0


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
        obs, _ = env.reset(jax.random.key(42))
        assert obs.shape == (4, 4, 4, 3)

    def test_step_batch_shape(self):
        env = VecEnv(_env(), num_envs=4)
        _, states = env.reset(jax.random.key(42))
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
        _, states = env.reset(jax.random.key(0))
        actions = jnp.zeros(1, dtype=jnp.int32)
        _, new_states, _, done, _ = env.step(states, actions)
        assert bool(done[0])
        assert int(new_states.step[0]) == 0

    def test_no_reset_when_not_done(self):
        env = VecEnv(_PixelEnv(config=EnvConfig(max_steps=100)), num_envs=1)
        _, states = env.reset(jax.random.key(0))
        actions = jnp.zeros(1, dtype=jnp.int32)
        _, new_states, _, done, _ = env.step(states, actions)
        assert not bool(done[0])
        assert int(new_states.step[0]) == 1

    def test_auto_reset_jit_compatible(self):
        env = VecEnv(_env(), num_envs=2)
        _, states = env.reset(jax.random.key(0))
        actions = jnp.zeros(2, dtype=jnp.int32)
        obs, _, _, _, _ = jax.jit(env.step)(states, actions)
        assert obs.shape == (2, 4, 4, 3)


class TestJitWrappedVecEnv:
    def test_jit_wrapper_around_vec_env(self):
        from envrax.wrappers import JitWrapper

        vec = VecEnv(_env(), num_envs=4)
        jit_vec = JitWrapper(vec, cache_dir=None)

        obs, states = jit_vec.reset(jax.random.key(0))
        assert obs.shape == (4, 4, 4, 3)

        actions = jnp.zeros(4, dtype=jnp.int32)
        obs, states, rewards, dones, _ = jit_vec.step(states, actions)
        assert obs.shape == (4, 4, 4, 3)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)


# ---------------------------------------------------------------------------
# RecordVideo tests
# ---------------------------------------------------------------------------


@chex.dataclass
class _RenderState(EnvState):
    pass


class _RenderEnv(JaxEnv[Box, Discrete, _RenderState, EnvConfig]):
    """Minimal env that implements render()."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=255, shape=(2, 2, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng):
        obs = jnp.full((2, 2, 3), 128, dtype=jnp.uint8)
        state = _RenderState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, state, action):
        import numpy as np

        new_state = state.__replace__(
            step=state.step + 1,
            done=jnp.bool_(state.step + 1 >= self.config.max_steps),
        )
        obs = jnp.full((2, 2, 3), 64, dtype=jnp.uint8)
        return obs, new_state, jnp.float32(1.0), new_state.done, {}

    def render(self, state):
        import numpy as np

        return np.full((2, 2, 3), 100, dtype=np.uint8)


class TestRecordVideo:
    def test_no_render_raises(self, tmp_path):
        import pytest
        from envrax.wrappers import RecordVideo

        with pytest.raises(TypeError, match="requires an environment that implements render"):
            RecordVideo(_env(), output_dir=tmp_path)

    def test_records_every_episode_by_default(self, tmp_path):
        from envrax.wrappers import RecordVideo

        env = RecordVideo(
            _RenderEnv(config=EnvConfig(max_steps=2)),
            output_dir=tmp_path,
        )
        _, state = env.reset(_RNG)
        assert env.recording is True

        _, state, _, _, _ = env.step(state, jnp.int32(0))
        _, state, _, done, _ = env.step(state, jnp.int32(0))
        assert bool(done)

        # MP4 should have been written
        mp4s = list(tmp_path.glob("*.mp4"))
        assert len(mp4s) == 1
        assert mp4s[0].name == "episode_0000.mp4"

    def test_episode_trigger(self, tmp_path):
        from envrax.wrappers import RecordVideo

        env = RecordVideo(
            _RenderEnv(config=EnvConfig(max_steps=1)),
            output_dir=tmp_path,
            episode_trigger=lambda ep: ep == 1,  # skip ep 0, record ep 1
        )

        # Episode 0 — should NOT record
        _, state = env.reset(_RNG)
        assert env.recording is False
        _, state, _, _, _ = env.step(state, jnp.int32(0))

        # Episode 1 — should record
        _, state = env.reset(_RNG)
        assert env.recording is True
        _, state, _, _, _ = env.step(state, jnp.int32(0))

        mp4s = list(tmp_path.glob("*.mp4"))
        assert len(mp4s) == 1
        assert mp4s[0].name == "episode_0001.mp4"

    def test_recording_trigger(self, tmp_path):
        from envrax.wrappers import RecordVideo

        should_record = False
        env = RecordVideo(
            _RenderEnv(config=EnvConfig(max_steps=1)),
            output_dir=tmp_path,
            recording_trigger=lambda: should_record,
        )

        # Trigger off — should NOT record
        _, state = env.reset(_RNG)
        assert env.recording is False
        _, state, _, _, _ = env.step(state, jnp.int32(0))

        # Trigger on — should record
        should_record = True
        _, state = env.reset(_RNG)
        assert env.recording is True
        _, state, _, _, _ = env.step(state, jnp.int32(0))

        mp4s = list(tmp_path.glob("*.mp4"))
        assert len(mp4s) == 1

    def test_step_trigger_mid_episode(self, tmp_path):
        from envrax.wrappers import RecordVideo

        env = RecordVideo(
            _RenderEnv(config=EnvConfig(max_steps=3)),
            output_dir=tmp_path,
            step_trigger=lambda s: s >= 2,  # start recording after step 2
        )

        _, state = env.reset(_RNG)
        assert env.recording is False  # no episode trigger, step trigger not fired yet

        _, state, _, _, _ = env.step(state, jnp.int32(0))  # step 1
        assert env.recording is False

        _, state, _, _, _ = env.step(state, jnp.int32(0))  # step 2 — trigger fires
        assert env.recording is True

        _, state, _, done, _ = env.step(state, jnp.int32(0))  # step 3 — done
        assert bool(done)

        mp4s = list(tmp_path.glob("*.mp4"))
        assert len(mp4s) == 1

    def test_recording_property(self, tmp_path):
        from envrax.wrappers import RecordVideo

        env = RecordVideo(
            _RenderEnv(config=EnvConfig(max_steps=1)),
            output_dir=tmp_path,
            episode_trigger=lambda ep: ep == 0,
        )

        _, state = env.reset(_RNG)
        assert env.recording is True

        _, state, _, _, _ = env.step(state, jnp.int32(0))
        # After done, recording resets
        assert env.recording is False


# ---------------------------------------------------------------------------
# require_box validation
# ---------------------------------------------------------------------------


class TestRequireBox:
    """`require_box` is the validator image-processing wrappers use at construction."""

    def test_non_box_observation_space_raises(self):
        # GrayscaleObservation requires Box — wrap an env with Discrete obs space
        import pytest

        from envrax.wrappers.utils import require_box

        @chex.dataclass
        class _S(EnvState):
            pass

        class _DiscreteObsEnv(JaxEnv[Discrete, Discrete, _S, EnvConfig]):
            @property
            def observation_space(self) -> Discrete:
                return Discrete(n=4)

            @property
            def action_space(self) -> Discrete:
                return Discrete(n=2)

            def reset(self, rng):
                return jnp.int32(0), _S(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))

            def step(self, state, action):
                return jnp.int32(0), state, jnp.float32(0.0), jnp.bool_(False), {}

        with pytest.raises(TypeError, match="requires a Box observation space"):
            require_box(_DiscreteObsEnv(), "TestWrapper")

    def test_wrong_rank_raises(self):
        import pytest

        from envrax.wrappers.utils import require_box

        # _PixelEnv is rank-3; require rank in (2,) → fails
        with pytest.raises(ValueError, match="requires observation rank in"):
            require_box(_env(), "TestWrapper", rank=2)

    def test_wrong_last_dim_raises(self):
        import pytest

        from envrax.wrappers.utils import require_box

        # _PixelEnv has last dim 3; require last_dim=1 → fails
        with pytest.raises(ValueError, match="requires last dim = 1"):
            require_box(_env(), "TestWrapper", last_dim=1)

    def test_wrong_dtype_raises(self):
        import pytest

        from envrax.wrappers.utils import require_box

        # _PixelEnv is uint8; require float32 → fails
        with pytest.raises(ValueError, match="requires float32 dtype"):
            require_box(_env(), "TestWrapper", dtype=jnp.float32)

    def test_passes_with_matching_constraints(self):
        from envrax.wrappers.utils import require_box

        space = require_box(
            _env(),
            "TestWrapper",
            rank=3,
            last_dim=3,
            dtype=jnp.uint8,
        )
        assert space.shape == (4, 4, 3)


# ---------------------------------------------------------------------------
# Wrapper factory mode (deferred construction)
# ---------------------------------------------------------------------------


class TestWrapperFactoryMode:
    """Calling a parameterised wrapper without `env` returns a deferred factory."""

    def test_deferred_factory_returns_wrapper_factory(self):
        from envrax.wrappers.base import _WrapperFactory

        factory = FrameStackObservation(n_stack=4)
        assert isinstance(factory, _WrapperFactory)

    def test_deferred_factory_constructs_wrapper_when_called(self):
        inner = ResizeObservation(GrayscaleObservation(_env()), h=4, w=4)
        factory = FrameStackObservation(n_stack=4)
        wrapped = factory(inner)
        assert isinstance(wrapped, FrameStackObservation)
        obs, _ = wrapped.reset(_RNG)
        assert obs.shape == (4, 4, 4)

    def test_deferred_factory_preserves_kwargs(self):
        inner = GrayscaleObservation(_env())
        factory = ResizeObservation(h=16, w=16)
        wrapped = factory(inner)
        assert wrapped.observation_space.shape == (16, 16)


# ---------------------------------------------------------------------------
# Wrapper.render forwards through the chain
# ---------------------------------------------------------------------------


class TestWrapperRender:
    def test_render_forwards_to_inner(self):
        # _RenderEnv defines render(); ClipReward is a pass-through wrapper
        env = ClipReward(_RenderEnv(config=_CONFIG))
        _, state = env.reset(_RNG)
        frame = env.render(state)
        assert frame.shape == (2, 2, 3)
        assert frame.dtype == jnp.uint8 or str(frame.dtype) == "uint8"

    def test_render_forwards_through_wrapper_chain(self):
        env = ExpandDims(ClipReward(_RenderEnv(config=_CONFIG)))
        _, state = env.reset(_RNG)
        frame = env.render(state)
        assert frame.shape == (2, 2, 3)

    def test_unwrapped_returns_innermost_env(self):
        inner = _RenderEnv(config=_CONFIG)
        env = ExpandDims(ClipReward(inner))
        assert env.unwrapped is inner


# ---------------------------------------------------------------------------
# VecEnv.render and __repr__
# ---------------------------------------------------------------------------


class TestVecEnvRenderAndRepr:
    def test_render_returns_single_frame_at_index(self):
        vec = VecEnv(_RenderEnv(config=_CONFIG), num_envs=4)
        _, states = vec.reset(_RNG)
        frame = vec.render(states, index=2)
        assert frame.shape == (2, 2, 3)

    def test_render_default_index_zero(self):
        vec = VecEnv(_RenderEnv(config=_CONFIG), num_envs=4)
        _, states = vec.reset(_RNG)
        frame = vec.render(states)
        assert frame.shape == (2, 2, 3)

    def test_repr_includes_inner_env_and_count(self):
        vec = VecEnv(_env(), num_envs=8)
        r = repr(vec)
        assert "VecEnv" in r
        assert "num_envs=8" in r
