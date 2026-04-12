import chex
import jax
import jax.numpy as jnp

from envrax.env import EnvConfig, EnvState, JaxEnv
from envrax.multi_env import MultiEnv
from envrax.spaces import Box, Discrete

# ---------------------------------------------------------------------------
# Two minimal envs with different obs shapes for heterogeneity testing
# ---------------------------------------------------------------------------


@chex.dataclass
class _VecState(EnvState):
    pass


class _VecEnv(JaxEnv[Box, Discrete, _VecState]):
    """float32[4] observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=(4,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        return obs, _VecState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))

    def step(self, state, action):
        s = state.__replace__(step=state.step + 1)
        return jnp.zeros((4,), jnp.float32), s, jnp.float32(1.0), jnp.bool_(False), {}


@chex.dataclass
class _PixState(EnvState):
    pass


class _PixEnv(JaxEnv[Box, Discrete, _PixState]):
    """uint8[4, 4, 3] observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=255, shape=(4, 4, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=3)

    def reset(self, rng):
        obs = jnp.full((4, 4, 3), 128, dtype=jnp.uint8)
        return obs, _PixState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))

    def step(self, state, action):
        s = state.__replace__(step=state.step + 1)
        return jnp.full((4, 4, 3), 64, jnp.uint8), s, jnp.float32(1.0), jnp.bool_(False), {}


_RNG = jax.random.key(0)
_CONFIG = EnvConfig(max_steps=10)


class TestMultiEnv:
    def test_num_envs(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        assert multi.num_envs == 2
        assert len(multi) == 2

    def test_envs_property(self):
        envs = [_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)]
        multi = MultiEnv(envs)
        assert multi.envs is envs

    def test_observation_spaces(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        spaces = multi.observation_spaces
        assert len(spaces) == 2
        assert spaces[0].shape == (4,)
        assert spaces[1].shape == (4, 4, 3)

    def test_action_spaces(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        spaces = multi.action_spaces
        assert spaces[0].n == 2
        assert spaces[1].n == 3

    def test_reset_returns_lists(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        obs_list, state_list = multi.reset(_RNG)
        assert isinstance(obs_list, list)
        assert isinstance(state_list, list)
        assert len(obs_list) == 2
        assert len(state_list) == 2

    def test_reset_obs_shapes_match_spaces(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        obs_list, _ = multi.reset(_RNG)
        assert obs_list[0].shape == (4,)
        assert obs_list[1].shape == (4, 4, 3)

    def test_reset_deterministic_same_seed(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG)])
        obs1, _ = multi.reset(_RNG)
        obs2, _ = multi.reset(_RNG)
        assert jnp.array_equal(obs1[0], obs2[0])

    def test_step_returns_correct_structure(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        _, states = multi.reset(_RNG)
        actions = [jnp.int32(0), jnp.int32(0)]
        obs, new_states, rewards, dones, infos = multi.step(states, actions)
        assert len(obs) == 2
        assert len(new_states) == 2
        assert len(rewards) == 2
        assert len(dones) == 2
        assert len(infos) == 2

    def test_step_obs_shapes(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        _, states = multi.reset(_RNG)
        actions = [jnp.int32(0), jnp.int32(0)]
        obs, _, _, _, _ = multi.step(states, actions)
        assert obs[0].shape == (4,)
        assert obs[1].shape == (4, 4, 3)

    def test_reset_at(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        obs, state = multi.reset_at(1, _RNG)
        assert obs.shape == (4, 4, 3)

    def test_step_at(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG)])
        _, states = multi.reset(_RNG)
        obs, new_state, reward, done, info = multi.step_at(0, states[0], jnp.int32(0))
        assert obs.shape == (4,)
        assert int(new_state.step) == 1

    def test_class_groups_same_class(self):
        multi = MultiEnv([
            _VecEnv(config=_CONFIG),
            _VecEnv(config=_CONFIG),
            _PixEnv(config=_CONFIG),
        ])
        groups = multi.class_groups
        assert "_VecEnv" in groups
        assert "_PixEnv" in groups
        assert groups["_VecEnv"] == [0, 1]
        assert groups["_PixEnv"] == [2]

    def test_empty_raises(self):
        import pytest

        with pytest.raises(ValueError, match="at least one"):
            MultiEnv([])

    def test_compile_with_jit_wrapped_envs(self):
        from envrax.wrappers import JitWrapper

        envs = [
            JitWrapper(_VecEnv(config=_CONFIG), cache_dir=None, pre_warm=False),
            JitWrapper(_PixEnv(config=_CONFIG), cache_dir=None, pre_warm=False),
        ]
        multi = MultiEnv(envs)
        multi.compile(progress=False)
        # Should work after compile
        obs_list, _ = multi.reset(_RNG)
        assert obs_list[0].shape == (4,)
        assert obs_list[1].shape == (4, 4, 3)

    def test_compile_skips_non_jit_envs(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG)])
        # Should not raise — silently skips non-JitWrapper envs
        multi.compile(progress=False)

    def test_repr(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        r = repr(multi)
        assert "MultiEnv" in r
        assert "_VecEnv" in r
        assert "_PixEnv" in r
